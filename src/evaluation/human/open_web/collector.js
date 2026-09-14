/* Server-side Apps Script. The builder appends OPEN_STUDY and PAGE_HTML.
 * Private helpers end in '_' so google.script.run cannot call them.
 */
var PARTICIPANTS = ['study_id','participant_id','nickname','token_hash','language','experience','consent_version','consent_at','created_at','last_seen_at','status','finish_reason','elapsed_ms','is_test','assignment_plan_json'];
var ASSIGNMENTS = ['study_id','participant_id','unit_token','position','assigned_at','is_test'];
var RATINGS = ['study_id','participant_id','unit_token','rating_id','position','chart_appropriateness','layout_quality','styling_accessibility','interaction_design','rationale_quality','overall_usefulness','comment','language','duration_ms','elapsed_ms','client_timestamp','received_at','rubric_version','is_test'];

function tables_(test) {
  var book = SpreadsheetApp.openById(OPEN_STUDY.spreadsheet_id);
  var prefix = (test ? 'test_' : '') + OPEN_STUDY.sheet_prefix;
  var result = {};
  [['participants',PARTICIPANTS],['assignments',ASSIGNMENTS],['ratings',RATINGS]].forEach(function(pair) {
    var sheet = book.getSheetByName(prefix + '_' + pair[0]);
    if (!sheet) { sheet = book.insertSheet(prefix + '_' + pair[0]); }
    if (!sheet.getLastRow()) {
      sheet.appendRow(pair[1]); sheet.setFrozenRows(1);
      sheet.getRange(1,1,1,pair[1].length).setFontWeight('bold').setBackground('#e8eef6');
    }
    var header = sheet.getRange(1,1,1,pair[1].length).getValues()[0];
    if (JSON.stringify(header) !== JSON.stringify(pair[1])) { throw new Error('SCHEMA_MISMATCH'); }
    result[pair[0]] = sheet;
  });
  return result;
}
function rows_(sheet, header) {
  if (sheet.getLastRow() < 2) { return []; }
  return sheet.getRange(2,1,sheet.getLastRow()-1,header.length).getValues().map(function(row,index) {
    var obj = {_row:index+2}; header.forEach(function(key,i){obj[key]=row[i];}); return obj;
  });
}
function text_(value, max) {
  var s = String(value || '').trim().slice(0,max);
  // Keep participant strings literal in spreadsheet cells (formula injection).
  return /^[=+@\-\t\r]/.test(s) ? "'" + s : s;
}
function write_(sheet, header, record, row) {
  sheet.getRange(row || sheet.getLastRow()+1,1,1,header.length)
    .setValues([header.map(function(key){return record[key] === undefined ? '' : record[key];})]);
}
function locked_(fn) {
  var lock=LockService.getScriptLock(); lock.waitLock(20000);
  try { return fn(); } finally { lock.releaseLock(); }
}
function hash_(value) {
  return Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256,value,Utilities.Charset.UTF_8)
    .map(function(n){return ('0'+((n+256)%256).toString(16)).slice(-2);}).join('');
}
function checkRequest_(request) {
  if (!request || request.study_id !== OPEN_STUDY.study_id) {throw new Error('WRONG_STUDY');}
  if (!/^[a-f0-9]{64}$/.test(request.token || '')) {throw new Error('INVALID_SESSION');}
}
function auth_(request, tables) {
  checkRequest_(request);
  var digest=hash_(request.token);
  var person=rows_(tables.participants,PARTICIPANTS).filter(function(p){return p.token_hash===digest;})[0];
  if (!person) {throw new Error('INVALID_SESSION');}
  return person;
}
function shuffle_(values) {
  var a=values.slice();
  for(var i=a.length-1;i>0;i--){var j=Math.floor(Math.random()*(i+1));var t=a[i];a[i]=a[j];a[j]=t;}
  return a;
}
function coverage_(ratings) {
  var counts={}; Object.keys(OPEN_STUDY.units).forEach(function(k){counts[k]=0;});
  var seen={}; ratings.forEach(function(r){
    var id=r.participant_id+'|'+r.unit_token;
    // A submitted page with only 'cannot judge' is not usable evidence.
    var usable=OPEN_STUDY.dimensions.some(function(k){return typeof r[k]==='number' && r[k]>=1 && r[k]<=5;});
    if (!seen[id] && usable && counts[r.unit_token]!==undefined) {counts[r.unit_token]++;seen[id]=true;}
  }); return counts;
}
function choosePacket_(tables) {
  var ratings=rows_(tables.ratings,RATINGS), counts=coverage_(ratings);
  if (Object.keys(counts).every(function(k){return counts[k]>=OPEN_STUDY.target_per_output;})) {return null;}
  var people=rows_(tables.participants,PARTICIPANTS), active={}, reservations={};
  people.forEach(function(p){
    if(p.status==='active' && Date.now()-Date.parse(p.last_seen_at)<86400000){active[p.participant_id]=true;}
  });
  var done={};ratings.forEach(function(r){done[r.participant_id+'|'+r.unit_token]=true;});
  rows_(tables.assignments,ASSIGNMENTS).forEach(function(a){
    if(active[a.participant_id]&&!done[a.participant_id+'|'+a.unit_token]){
      reservations[a.unit_token]=(reservations[a.unit_token]||0)+1;
    }
  });
  // Four complementary packets: one output per brief, two outputs per method.
  // Selection uses counts only; score values never decide assignment or stopping.
  var candidates=shuffle_(OPEN_STUDY.packets).map(function(packet){
    var cost=packet.reduce(function(total,k){
      return total+Math.pow(counts[k]+(reservations[k]||0),2);
    },0);
    return {packet:packet,cost:cost};
  });
  candidates.sort(function(a,b){return a.cost-b.cost;});
  return shuffle_(candidates[0].packet);
}
function session_(person,tables) {
  var assignments=rows_(tables.assignments,ASSIGNMENTS).filter(function(a){return a.participant_id===person.participant_id;});
  assignments.sort(function(a,b){return a.position-b.position;});
  var ratings=rows_(tables.ratings,RATINGS).filter(function(r){return r.participant_id===person.participant_id;});
  if(ratings.length===OPEN_STUDY.session_size && person.status!=='complete'){
    person.status='complete';person.finish_reason='all_rated';
    write_(tables.participants,PARTICIPANTS,person,person._row);
  }
  return {participant_id:person.participant_id,nickname:person.nickname,language:person.language,status:person.status,
    elapsed_ms:Number(person.elapsed_ms)||0,done:ratings.map(function(r){return r.unit_token;}),
    records:ratings.map(function(r){var scores={};OPEN_STUDY.dimensions.forEach(function(k){scores[k]=r[k]===''?null:r[k];});
      return {unit_token:r.unit_token,scores:scores,comment:r.comment,language:r.language,duration_ms:r.duration_ms,elapsed_ms:r.elapsed_ms,received_at:r.received_at};}),
    tasks:assignments.map(function(a){var unit=OPEN_STUDY.units[a.unit_token];return {
      unit_token:a.unit_token,position:a.position,brief_html:unit.brief_html,output_html:unit.output_html};})};
}
function ensureAssignments_(person,tables) {
  // Recover a partial Sheets write after a network/service failure without
  // assigning a new packet or leaving a participant with zero tasks.
  var packet=JSON.parse(person.assignment_plan_json);
  var existing=rows_(tables.assignments,ASSIGNMENTS).filter(function(a){return a.participant_id===person.participant_id;});
  var missing=[];packet.forEach(function(k,i){
    if(!existing.some(function(a){return a.unit_token===k;})){
      missing.push([OPEN_STUDY.study_id,person.participant_id,k,i+1,person.created_at,person.is_test]);
    }
  });
  if(missing.length){tables.assignments.getRange(tables.assignments.getLastRow()+1,1,missing.length,ASSIGNMENTS.length).setValues(missing);}
}
function registerParticipant(request) {
  return locked_(function(){
    checkRequest_(request);
    var tables=tables_(request.is_test===true), digest=hash_(request.token);
    var previous=rows_(tables.participants,PARTICIPANTS).filter(function(p){return p.token_hash===digest;})[0];
    if(previous){ensureAssignments_(previous,tables);return session_(previous,tables);}
    if(request.consent!==true || request.consent_version!==OPEN_STUDY.consent_version){throw new Error('CONSENT_REQUIRED');}
    var nickname=text_(request.nickname,40);
    if(!nickname){throw new Error('NAME_REQUIRED');}
    if(['de','en'].indexOf(request.language)<0){throw new Error('INVALID_LANGUAGE');}
    if(['none','some','experienced','unspecified'].indexOf(request.experience)<0){throw new Error('INVALID_EXPERIENCE');}
    var packet=choosePacket_(tables);
    if(!packet){return {closed:true};}
    var now=new Date().toISOString();
    var person={study_id:OPEN_STUDY.study_id,participant_id:'p_'+Utilities.getUuid(),nickname:nickname,
      token_hash:digest,language:request.language,experience:request.experience,consent_version:OPEN_STUDY.consent_version,
      consent_at:now,created_at:now,last_seen_at:now,status:'active',elapsed_ms:0,is_test:request.is_test===true,
      assignment_plan_json:JSON.stringify(packet)};
    write_(tables.participants,PARTICIPANTS,person);
    person._row=tables.participants.getLastRow();
    ensureAssignments_(person,tables);
    SpreadsheetApp.flush();
    return session_(person,tables);
  });
}
function resumeParticipant(request) {
  return locked_(function(){
    var tables=tables_(request.is_test===true),person=auth_(request,tables);
    ensureAssignments_(person,tables);
    person.last_seen_at=new Date().toISOString();write_(tables.participants,PARTICIPANTS,person,person._row);
    return session_(person,tables);
  });
}
function submitRating(request) {
  return locked_(function(){
    var tables=tables_(request.is_test===true),person=auth_(request,tables),rating=request.rating || {};
    var assigned=rows_(tables.assignments,ASSIGNMENTS).filter(function(a){return a.participant_id===person.participant_id&&a.unit_token===rating.unit_token;})[0];
    if(!assigned){throw new Error('NOT_ASSIGNED');}
    var existing=rows_(tables.ratings,RATINGS).filter(function(r){return r.participant_id===person.participant_id&&r.unit_token===rating.unit_token;})[0];
    // Idempotent acknowledgment even when a response was lost after final submission.
    if(existing){return {saved:true,duplicate:true,unit_token:rating.unit_token};}
    if(person.status!=='active'){throw new Error('SESSION_FINISHED');}
    var scores=rating.scores || {};
    if(Object.keys(scores).length!==OPEN_STUDY.dimensions.length){throw new Error('INVALID_SCORES');}
    OPEN_STUDY.dimensions.forEach(function(k){
      if(!Object.prototype.hasOwnProperty.call(scores,k) || !(scores[k]===null || (Number.isInteger(scores[k])&&scores[k]>=1&&scores[k]<=5))){throw new Error('INVALID_SCORES');}
    });
    if(!Number.isFinite(rating.duration_ms)||rating.duration_ms<0||rating.duration_ms>1800000){throw new Error('INVALID_DURATION');}
    if(!Number.isFinite(rating.elapsed_ms)||rating.elapsed_ms<0||rating.elapsed_ms>1800000){throw new Error('TIME_LIMIT');}
    if(rating.duration_ms>rating.elapsed_ms){throw new Error('INVALID_DURATION');}
    if(['de','en'].indexOf(rating.language)<0){throw new Error('INVALID_LANGUAGE');}
    var now=new Date().toISOString(),row={study_id:OPEN_STUDY.study_id,participant_id:person.participant_id,
      unit_token:rating.unit_token,rating_id:person.participant_id+'|'+rating.unit_token,position:assigned.position,
      comment:text_(rating.comment,500),language:rating.language,duration_ms:Math.round(rating.duration_ms),
      elapsed_ms:Math.round(rating.elapsed_ms),client_timestamp:text_(rating.client_timestamp,40),received_at:now,
      rubric_version:OPEN_STUDY.rubric_version,is_test:request.is_test===true};
    OPEN_STUDY.dimensions.forEach(function(k){row[k]=scores[k]===null?'':scores[k];});
    write_(tables.ratings,RATINGS,row);
    person.last_seen_at=now;person.elapsed_ms=Math.max(Number(person.elapsed_ms)||0,rating.elapsed_ms);
    var count=rows_(tables.ratings,RATINGS).filter(function(r){return r.participant_id===person.participant_id;}).length;
    if(count===OPEN_STUDY.session_size){person.status='complete';person.finish_reason='all_rated';}
    write_(tables.participants,PARTICIPANTS,person,person._row);
    SpreadsheetApp.flush();
    return {saved:true,unit_token:rating.unit_token};
  });
}
function finishParticipation(request) {
  return locked_(function(){
    var tables=tables_(request.is_test===true),person=auth_(request,tables);
    if(person.status==='active'){
      person.status='partial';person.finish_reason=request.reason==='time_limit'?'time_limit':'voluntary_stop';
      person.elapsed_ms=Math.max(Number(person.elapsed_ms)||0,Math.min(1800000,Math.max(0,Number(request.elapsed_ms)||0)));
      person.last_seen_at=new Date().toISOString();write_(tables.participants,PARTICIPANTS,person,person._row);
    }
    return {finished:true};
  });
}
function doGet(e) {
  if(e && e.parameter && e.parameter.health==='1'){
    return ContentService.createTextOutput(JSON.stringify({ok:true,version:OPEN_STUDY.study_id})).setMimeType(ContentService.MimeType.JSON);
  }
  var boot={study_id:OPEN_STUDY.study_id,consent_version:OPEN_STUDY.consent_version,
    session_size:OPEN_STUDY.session_size,is_test:Boolean(e && e.parameter && e.parameter.demo==='1')};
  return HtmlService.createHtmlOutput(PAGE_HTML.replace('/*BOOT*/',JSON.stringify(boot)))
    .setTitle('Dashboard study | Dashboard-Studie').addMetaTag('viewport','width=device-width, initial-scale=1');
}
function doPost() {
  return ContentService.createTextOutput(JSON.stringify({ok:false,error:'LEGACY_COLLECTION_CLOSED'})).setMimeType(ContentService.MimeType.JSON);
}
