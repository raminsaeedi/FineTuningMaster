/* Run with node; Apps Script service doubles, no network or real research data. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto');

function fixture() {
  let held=false, acquisitions=0;
  const sheets={};
  class Sheet {
    constructor(){this.data=[];}
    getLastRow(){return this.data.length;}
    appendRow(row){this.data.push([...row]);}
    setFrozenRows(){}
    getRange(r,c,n,m){const self=this;return {
      getValues(){return Array.from({length:n},(_,i)=>Array.from({length:m},(_,j)=>(self.data[r+i-1]||[])[c+j-1]??''));},
      setValues(values){assert.equal(values.length,n);for(let i=0;i<n;i++){self.data[r+i-1]??=[];for(let j=0;j<m;j++)self.data[r+i-1][c+j-1]=values[i][j];}return this;},
      setFontWeight(){return this;},setBackground(){return this;}
    };}
  }
  const dimensions=['chart_appropriateness','layout_quality','styling_accessibility','interaction_design','rationale_quality','overall_usefulness'];
  const units={};for(let i=0;i<8;i++)for(let m=0;m<4;m++)units['u'+i+m]={brief_html:'brief'+i,output_html:'output'+m};
  const packets=Array.from({length:4},(_,s)=>Array.from({length:8},(_,i)=>'u'+i+((i+s)%4)));
  const config={study_id:'test-study',sheet_prefix:'openv1',spreadsheet_id:'fake',dimensions,units,packets,session_size:8,target_per_output:3,consent_version:'v1',rubric_version:'r2'};
  const ctx={console,Date,Math,Number,JSON,OPEN_STUDY:config,PAGE_HTML:'/*BOOT*/',
    SpreadsheetApp:{openById(){return {getSheetByName:n=>sheets[n],insertSheet:n=>(sheets[n]=new Sheet())};},flush(){}},
    LockService:{getScriptLock(){return {waitLock(){assert(!held);held=true;acquisitions++;},releaseLock(){held=false;}};}},
    Utilities:{getUuid:crypto.randomUUID,DigestAlgorithm:{SHA_256:'sha'},Charset:{UTF_8:'utf8'},computeDigest(_a,value){return [...crypto.createHash('sha256').update(value).digest()];}}
  };
  vm.createContext(ctx);vm.runInContext(fs.readFileSync('src/evaluation/human/open_web/collector.js','utf8'),ctx);
  function request(i=1,test=false){return {study_id:config.study_id,token:i.toString(16).padStart(64,'0'),is_test:test,consent:true,consent_version:'v1',nickname:'same nickname',language:'de',experience:'unspecified'};}
  function rating(req,unit,value=4){return {...req,rating:{unit_token:unit,scores:Object.fromEntries(dimensions.map(k=>[k,value])),language:'en',duration_ms:1000,elapsed_ms:2000,comment:'=malicious()',client_timestamp:'2026-09-14T00:00:00Z'}};}
  return {ctx,sheets,request,rating,config,locked:()=>held,acquisitions:()=>acquisitions};
}

{
  const {ctx,request,rating,sheets,locked}=fixture(),r=request();
  assert.throws(()=>ctx.registerParticipant({...r,consent:false}),/CONSENT/);
  assert(!locked());
  const p=ctx.registerParticipant(r);
  assert.equal(p.tasks.length,8);
  assert.equal(new Set(p.tasks.map(t=>t.brief_html)).size,8);
  const methods=p.tasks.map(t=>t.unit_token.at(-1));
  for(const m of ['0','1','2','3'])assert.equal(methods.filter(x=>x===m).length,2);
  assert.equal(ctx.registerParticipant(r).participant_id,p.participant_id);
  assert.notEqual(ctx.registerParticipant(request(2)).participant_id,p.participant_id);
  assert(!JSON.stringify(p).includes('token_hash'));
  assert(!JSON.stringify(sheets.openv1_participants.data).includes(r.token));
  const u=p.tasks[0].unit_token;
  assert.throws(()=>ctx.resumeParticipant(request(300)),/INVALID_SESSION/);
  assert.throws(()=>ctx.submitRating(rating(r,'unknown')),/NOT_ASSIGNED/);
  assert.throws(()=>ctx.submitRating(rating(r,u,0)),/INVALID_SCORES/);
  assert.throws(()=>ctx.submitRating(rating(r,u,true)),/INVALID_SCORES/);
  const late=rating(r,u);late.rating.elapsed_ms=1800001;
  assert.throws(()=>ctx.submitRating(late),/TIME_LIMIT/);
  ctx.submitRating(rating(r,u,null));
  assert.equal(ctx.submitRating(rating(r,u,4)).duplicate,true);
  assert.equal(sheets.openv1_ratings.getLastRow(),2);
  const saved=ctx.rows_(sheets.openv1_ratings,ctx.RATINGS)[0];
  assert.equal(saved.comment,"'=malicious()");
  assert.equal(ctx.coverage_([saved])[u],0);
  assert.equal(ctx.resumeParticipant(r).records[0].scores.overall_usefulness,null);
  ctx.finishParticipation({...r,reason:'time_limit',elapsed_ms:1800000});
  assert.equal(ctx.resumeParticipant(r).status,'partial');
  assert.throws(()=>ctx.submitRating(rating(r,p.tasks[1].unit_token)),/SESSION_FINISHED/);
  assert(!locked());
}
{
  const {ctx,request,rating,sheets,config}=fixture();
  // Four simultaneous reservations cover all outputs exactly once, then repeat.
  const people=Array.from({length:12},(_,i)=>ctx.registerParticipant(request(i+1)));
  for(let block=0;block<3;block++)assert.equal(new Set(people.slice(block*4,block*4+4).flatMap(p=>p.tasks.map(t=>t.unit_token))).size,32);
  for(let i=0;i<12;i++)for(const task of people[i].tasks)ctx.submitRating(rating(request(i+1),task.unit_token));
  assert.equal(sheets.openv1_ratings.getLastRow(),97);
  assert.equal(ctx.resumeParticipant(request(1)).status,'complete');
  assert.equal(ctx.registerParticipant(request(99)).closed,true);
  assert(Object.values(ctx.coverage_(ctx.rows_(sheets.openv1_ratings,ctx.RATINGS))).every(c=>c===3));
  const demo=ctx.registerParticipant(request(1,true));
  assert.notEqual(demo.participant_id,people[0].participant_id);
  ctx.submitRating(rating(request(1,true),demo.tasks[0].unit_token));
  assert.equal(sheets.test_openv1_ratings.getLastRow(),2);
  assert.equal(sheets.openv1_ratings.getLastRow(),97);
  assert.equal(ctx.submitRating(rating(request(1),people[0].tasks[7].unit_token)).duplicate,true);
}
{
  const {ctx,request,sheets}=fixture(),r=request();
  const p=ctx.registerParticipant(r);
  // A Sheets write was interrupted after only four assignment rows.
  sheets.openv1_assignments.data=sheets.openv1_assignments.data.slice(0,5);
  const recovered=ctx.registerParticipant(r);
  assert.equal(recovered.participant_id,p.participant_id);
  assert.equal(recovered.tasks.length,8);
  assert.equal(sheets.openv1_participants.getLastRow(),2);
}
console.log('PASS: allocation, consent, authentication, idempotence, missing scores, cap, dropout, 12-session coverage, demo isolation, partial-write recovery');
