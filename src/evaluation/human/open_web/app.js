(function(){
  'use strict';
  const B=window.BOOT, $=id=>document.getElementById(id), key='open-eval:'+B.study_id+(B.is_test?':test':'');
  const DIM=['chart_appropriateness','layout_quality','styling_accessibility','interaction_design','rationale_quality','overall_usefulness'];
  const Q={
    de:[['Diagramme','Passen die vorgeschlagenen Diagramme zur Aufgabe?','1: passen kaum · 3: teilweise passend · 5: sehr passend'],['Anordnung','Ist klar, wo Informationen stehen und was wichtig ist?','1: unklar · 3: brauchbar · 5: klar und gut geordnet'],['Lesbarkeit','Sind Farben, Beschriftungen und Kontraste gut lesbar?','1: kaum berücksichtigt · 3: teilweise · 5: gut berücksichtigt'],['Bedienung','Helfen die vorgeschlagenen Filter und Funktionen bei der Aufgabe?','1: fehlen oder stören · 3: teilweise hilfreich · 5: sehr hilfreich'],['Begründung','Erklärt der Text verständlich und überzeugend, warum der Vorschlag passt?','1: fehlt oder unpassend · 3: plausibel, aber allgemein · 5: konkret und überzeugend'],['Gesamtnutzen','Wie hilfreich ist der ganze Vorschlag für die beschriebene Aufgabe?','1: kaum hilfreich · 3: teilweise hilfreich · 5: sehr hilfreich']],
    en:[['Charts','Do the suggested charts fit the task?','1: poor fit · 3: partly suitable · 5: very suitable'],['Layout','Is it clear where information goes and what matters most?','1: unclear · 3: usable · 5: clear and well organised'],['Readability','Are colours, labels and contrast easy to read?','1: hardly addressed · 3: partly addressed · 5: well addressed'],['Controls','Do the suggested filters and controls help with the task?','1: missing or unhelpful · 3: partly helpful · 5: very helpful'],['Explanation','Does the text clearly and convincingly explain why the design fits?','1: missing or unsuitable · 3: plausible but general · 5: specific and convincing'],['Overall usefulness','How useful is the whole recommendation for the stated task?','1: hardly useful · 3: partly useful · 5: very useful']]
  };
  const T={
    de:{institution:'Ruhrwest University · Masterarbeit',title:'Wie hilfreich ist dieser Dashboard-Vorschlag?',lead:'Hilf uns, KI-Vorschläge für Diagramme zu bewerten. Deine persönliche Einschätzung zählt.',workload:'8 kurze Aufgaben',time:'ca. 15–25 Minuten',install:'Ohne Installation',how:'So funktioniert es',step1:'Lies die Aufgabe und den Vorschlag.',step2:'Beantworte sechs kurze Fragen mit 1 bis 5.',step3:'Speichern. Weiter. Fertig.',contentLanguage:'Aufgaben und KI-Texte bleiben auf Englisch. Du solltest einfache englische Texte verstehen.',ratingRule:'Bewerte nur, was dort steht. Fehlende Angaben dürfen eine niedrige Note bekommen. Wenn du etwas nicht beurteilen kannst, wähle „Kann ich nicht beurteilen“.',exampleTitle:'Ein Beispiel und kurze Begriffshilfe',example:'Die Aufgabe verlangt einen Vergleich zwischen Städten. Ein klar erklärtes Balkendiagramm kann gut passen. Ein Vorschlag für eine andere Frage passt schlechter. Bewerte die übrigen Fragen getrennt.',glossary:'KPI = Kennzahl. Layout = Anordnung. Styling = Farben und Lesbarkeit. Interaction = zum Beispiel Filter oder Informationen beim Darüberfahren.',privacyTitle:'Freiwillig und vertraulich',privacy:'Wir speichern Wunschname, zufällige ID, Antworten, Sprache und Zeiten für bis zu 12 Monate. Namen bleiben in einer separaten, privaten Tabelle. Die Masterarbeit verwendet zusammengefasste Ergebnisse. Bitte keine echten Namen oder Kontaktdaten in Kommentare schreiben.',hosting:'Google verarbeitet technische Verbindungsdaten nach seiner Datenschutzerklärung. Du kannst jederzeit aufhören. Nach 30 aktiven Minuten endet die Sitzung automatisch; vorhandene Antworten bleiben gespeichert.',withdraw:'Fragen oder Rückzug vor der endgültigen Auswertung: Sende deine ID an',name:'Dein Wunschname',nicknameHelp:'Ein erfundener Name reicht. Gleiche Namen bekommen verschiedene IDs.',experience:'Erfahrung mit Diagrammen (freiwillig)',unspecified:'Keine Angabe',none:'Kaum Erfahrung',some:'Etwas Erfahrung',experienced:'Viel Erfahrung',consent:'Ich habe die Hinweise gelesen und nehme freiwillig teil. Ich verstehe die englischen Aufgaben und nehme nur einmal teil.',start:'Teilnehmen',resumeTitle:'Schon angefangen? Mit meinem Code fortsetzen',codeLabel:'Dein persönlicher Fortsetzungscode',recover:'Fortsetzen',codeHelp:'Notiere diesen Code. Damit kannst du später fortsetzen. Teile ihn nicht mit anderen.',copy:'Code kopieren',stepBrief:'1 · Die Aufgabe',stepOutput:'2 · Der KI-Vorschlag',yourRating:'3 · Deine Einschätzung',independent:'Jede Frage einzeln bewerten. Keine Antwort ist vorausgewählt.',commentLabel:'Kommentar (freiwillig)',saveNext:'Speichern und weiter',pause:'Pause',stop:'Teilnahme beenden',pausedTitle:'Pause. Dein Fortschritt bleibt erhalten.',pausedText:'Du kannst diesen Tab schließen oder später weitermachen.',thanks:'Vielen Dank',withdrawShort:'Für einen Rückzug vor der Auswertung sende deine ID an die Kontaktadresse unten.',backup:'Meine Antworten herunterladen',na:'Kann ich nicht beurteilen',labels:['Schwach','Eher schwach','Mittel','Gut','Sehr gut'],task:'Aufgabe',of:'von',saved:'Gespeichert',saving:'Wird gespeichert…',missing:'Bitte beantworte jede Frage. „Kann ich nicht beurteilen“ ist auch möglich.',network:'Verbindung fehlgeschlagen. Deine Eingaben bleiben hier erhalten. Bitte erneut versuchen.',storage:'Dieser Browser kann Fortschritt nicht dauerhaft speichern. Notiere deinen Code und lass diesen Tab geöffnet.',closed:'Die Studie hat ihr Erhebungsziel erreicht. Vielen Dank für dein Interesse.',doneTitle:'Deine Antworten sind gespeichert.',partialTitle:'Deine Teilnahme ist beendet.',doneText:'Danke für deine Zeit! Du kannst diese Seite schließen.',partialText:'Vorhandene Antworten sind gespeichert. Unbeantwortete Fragen bleiben leer.',id:'Deine ID',timeLimit:'30 aktive Minuten sind erreicht. Deine bisherigen Antworten bleiben gespeichert.',copied:'Code kopiert.',invalid:'Der Code passt nicht zu dieser Studie. Prüfe den Code oder kontaktiere die Studienleitung.',consentError:'Bitte gib einen Wunschnamen ein und bestätige die Teilnahme.',onlyOnce:'Auf diesem Browser besteht bereits eine Teilnahme. Wir setzen sie fort.'},
    en:{institution:'Ruhrwest University · Master’s thesis',title:'How useful is this dashboard recommendation?',lead:'Help us evaluate AI suggestions for charts. We want your personal assessment.',workload:'8 short tasks',time:'about 15–25 minutes',install:'No installation',how:'How it works',step1:'Read the task and the recommendation.',step2:'Answer six short questions with a score from 1 to 5.',step3:'Save. Next. Done.',contentLanguage:'Tasks and AI texts are in English. You should be comfortable reading simple English.',ratingRule:'Judge only what is written. Missing information may deserve a low score. If you cannot judge something, choose “Cannot judge”.',exampleTitle:'An example and a few useful terms',example:'The task asks to compare cities. A clearly explained bar chart may fit well. A recommendation for a different question fits less well. Rate the other questions separately.',glossary:'KPI = a measure of interest. Layout = arrangement. Styling = colours and readability. Interaction = for example filters or information shown on hover.',privacyTitle:'Voluntary and confidential',privacy:'We store your nickname, random ID, answers, language and timings for up to 12 months. Nicknames stay in a separate private table. The thesis uses aggregated results. Please do not include real names or contact details in comments.',hosting:'Google processes technical connection data under its privacy policy. You can stop at any time. The session ends after 30 active minutes; answers already submitted stay saved.',withdraw:'Questions or withdrawal before final analysis: send your ID to',name:'Your nickname',nicknameHelp:'An invented name is fine. Matching names still receive different IDs.',experience:'Experience with charts (optional)',unspecified:'Prefer not to say',none:'Little experience',some:'Some experience',experienced:'A lot of experience',consent:'I have read the information and agree to take part voluntarily. I understand the English tasks and will participate only once.',start:'Take part',resumeTitle:'Already started? Continue with my code',codeLabel:'Your personal continuation code',recover:'Continue',codeHelp:'Keep this code to continue later. Do not share it with other people.',copy:'Copy code',stepBrief:'1 · The task',stepOutput:'2 · The AI recommendation',yourRating:'3 · Your assessment',independent:'Rate each question separately. No answer is preselected.',commentLabel:'Comment (optional)',saveNext:'Save and continue',pause:'Pause',stop:'End participation',pausedTitle:'Paused. Your progress is kept.',pausedText:'You may close this tab or continue later.',thanks:'Thank you',withdrawShort:'To withdraw before analysis, send your ID to the contact address below.',backup:'Download my answers',na:'Cannot judge',labels:['Poor','Weak','Moderate','Good','Very good'],task:'Task',of:'of',saved:'Saved',saving:'Saving…',missing:'Please answer each question. You can also select “Cannot judge”.',network:'Connection failed. Your entries are kept here. Please try again.',storage:'This browser cannot keep progress after you close it. Write down your code and keep this tab open.',closed:'The study has reached its collection target. Thank you for your interest.',doneTitle:'Your answers are saved.',partialTitle:'Your participation has ended.',doneText:'Thank you for your time! You can close this page.',partialText:'Submitted answers are saved. Unanswered questions stay empty.',id:'Your ID',timeLimit:'You have reached 30 active minutes. Your submitted answers are saved.',copied:'Code copied.',invalid:'This code does not match this study. Check it or contact the researcher.',consentError:'Please enter a nickname and agree to participate.',onlyOnce:'This browser already has a participation. We will continue it.'}
  };
  let state={token:null,session:null,elapsed:0,duration:0,drafts:{},records:[],lang:'de'},view='intro',busy=false,last=Date.now(),beforePause='rating';
  try{state=Object.assign(state,JSON.parse(localStorage.getItem(key)||'{}'));}catch(e){}
  if(!['de','en'].includes(state.lang))state.lang='de';
  const t=k=>T[state.lang][k];
  function persist(){try{localStorage.setItem(key,JSON.stringify(state));}catch(e){message('notice',t('storage'));}}
  function message(id,text){$(id).textContent=text;$(id).classList.toggle('hidden',!text);}
  function show(name){view=name;['intro','rating','paused','done'].forEach(k=>$(k).classList.toggle('hidden',k!==name));last=Date.now();window.scrollTo(0,0);}
  function rpc(method,extra){return new Promise((resolve,reject)=>{
    google.script.run.withSuccessHandler(resolve).withFailureHandler(reject)[method](Object.assign({study_id:B.study_id,token:state.token,is_test:B.is_test},extra||{}));
  });}
  function token(){return Array.from(crypto.getRandomValues(new Uint8Array(32)),b=>b.toString(16).padStart(2,'0')).join('');}
  function task(){return state.session&&state.session.tasks.find(x=>!state.session.done.includes(x.unit_token));}
  function translate(){
    document.documentElement.lang=state.lang;
    document.querySelectorAll('[data-t]').forEach(el=>{el.textContent=t(el.dataset.t);});
    ['de','en'].forEach(lang=>$(lang).setAttribute('aria-pressed',String(lang===state.lang)));
    $('scale-guide').innerHTML=t('labels').map((s,i)=>'<span><b>'+(i+1)+'</b>'+s+'</span>').join('');
    if(state.session){$('identity-label').textContent=t('id')+': '+state.session.participant_id;$('identity-code').textContent=state.token;}
    if(view==='rating')renderQuestions();
    if(view==='done')done(false);
  }
  function draft(){const current=task();if(!current)return null;return state.drafts[current.unit_token]||(state.drafts[current.unit_token]={scores:{},comment:'',duration:0});}
  function renderQuestions(){
    const d=draft();if(!d)return;
    $('questions').innerHTML=DIM.map((key,i)=>{
      const q=Q[state.lang][i];
      return '<fieldset class="question" id="q-'+key+'"><legend>'+(i+1)+'. '+q[0]+' · '+q[1]+'</legend><p class="help">'+q[2]+'</p><div class="choices">'+t('labels').map((label,j)=>'<label class="choice"><input type="radio" name="'+key+'" value="'+(j+1)+'"'+(d.scores[key]===j+1?' checked':'')+'><span>'+(j+1)+' · '+label+'</span></label>').join('')+'</div><label class="choice na"><input type="radio" name="'+key+'" value="na"'+(d.scores[key]===null?' checked':'')+'><span>'+t('na')+'</span></label></fieldset>';
    }).join('');
    $('comment').value=d.comment;
  }
  function render(){
    if(state.session.status!=='active'||!task()){done();return;}
    show('rating');translate();const current=task();
    $('progress-text').textContent=t('task')+' '+current.position+' '+t('of')+' '+state.session.tasks.length;
    $('progress').max=state.session.tasks.length;$('progress').value=state.session.done.length;
    $('brief').innerHTML=current.brief_html;$('output').innerHTML=current.output_html;
    $('saved').textContent=state.session.done.length?t('saved'):'';
    state.duration=draft().duration||0;persist();
  }
  function done(navigate=true){
    if(navigate)show('done');
    const complete=state.session&&state.session.done.length===B.session_size;
    $('done-title').textContent=t(complete?'doneTitle':'partialTitle');
    $('done-text').textContent=t(complete?'doneText':'partialText');
    $('done-id').textContent=t('id')+': '+(state.session?state.session.participant_id:'');
  }
  async function resume(){
    if(!state.token)return;
    const session=await rpc('resumeParticipant');state.session=session;state.records=session.records||[];state.elapsed=Math.max(state.elapsed,session.elapsed_ms||0);persist();
    if(state.elapsed>=1800000&&session.status==='active'){await finish('time_limit',true);return;}
    render();
  }
  async function finish(reason,internal=false){
    if(busy&&!internal)return;busy=true;setDisabled(true);
    try{await rpc('finishParticipation',{reason:reason,elapsed_ms:Math.min(1800000,state.elapsed)});
      if(state.session.status!=='complete')state.session.status='partial';persist();done();
      if(reason==='time_limit')message('notice',t('timeLimit'));
    }catch(e){show('paused');message('error',t('network'));}finally{busy=false;setDisabled(false);}
  }
  function setDisabled(value){['start','submit','recover','stop','pause','continue'].forEach(id=>$(id).disabled=value);if(state.elapsed>=1800000)$('start').disabled=true;}
  $('join-form').addEventListener('submit',async e=>{
    e.preventDefault();if(busy)return;
    if(state.elapsed>=1800000){message('notice',t('timeLimit'));return;}
    if(!$('nickname').value.trim()||!$('consent').checked){message('error',t('consentError'));return;}
    busy=true;setDisabled(true);message('error','');state.token=state.token||token();persist();
    try{
      const result=await rpc('registerParticipant',{nickname:$('nickname').value.trim(),experience:$('experience').value,
        language:state.lang,consent:true,consent_version:B.consent_version});
      if(result.closed){message('notice',t('closed'));return;}
      state.session=result;state.records=result.records||[];state.elapsed=Math.max(state.elapsed,result.elapsed_ms||0);persist();render();
    }catch(e){message('error',t('network'));}finally{busy=false;setDisabled(false);}
  });
  $('rating-form').addEventListener('change',e=>{
    const d=draft();if(!d)return;
    if(e.target.type==='radio'){d.scores[e.target.name]=e.target.value==='na'?null:Number(e.target.value);$('q-'+e.target.name).classList.remove('missing');persist();}
  });
  $('comment').addEventListener('input',()=>{if(draft()){draft().comment=$('comment').value;persist();}});
  $('rating-form').addEventListener('submit',async e=>{
    e.preventDefault();if(busy)return;const d=draft(),current=task();if(!d||!current)return;
    const missing=DIM.filter(k=>!Object.prototype.hasOwnProperty.call(d.scores,k));
    if(missing.length){missing.forEach(k=>$('q-'+k).classList.add('missing'));message('error',t('missing'));$('q-'+missing[0]).scrollIntoView({block:'center'});return;}
    if(state.elapsed>=1800000){await finish('time_limit');return;}
    busy=true;setDisabled(true);message('error','');$('saved').textContent=t('saving');persist();
    const rating={unit_token:current.unit_token,scores:d.scores,comment:d.comment,language:state.lang,
      duration_ms:Math.min(1800000,Math.round(state.duration)),elapsed_ms:Math.min(1800000,Math.round(state.elapsed)),client_timestamp:new Date().toISOString()};
    try{const result=await rpc('submitRating',{rating});if(!result.saved)throw new Error('SAVE_FAILED');
      state.records.push(rating);state.session.done.push(current.unit_token);delete state.drafts[current.unit_token];
      state.duration=0;persist();render();
    }catch(e){message('error',t('network'));$('saved').textContent='';}finally{busy=false;setDisabled(false);}
  });
  ['de','en'].forEach(lang=>$(lang).addEventListener('click',()=>{state.lang=lang;persist();translate();}));
  $('pause').addEventListener('click',()=>{beforePause=view;persist();show('paused');});
  $('continue').addEventListener('click',()=>{message('error','');if(state.elapsed>=1800000){finish('time_limit');}else if(beforePause==='rating'){render();}else{show('intro');}});
  $('stop').addEventListener('click',()=>finish('voluntary_stop'));
  $('recover').addEventListener('click',async()=>{
    if(busy)return;const candidate=$('recovery-code').value.trim();
    if(!/^[a-f0-9]{64}$/.test(candidate)){message('error',t('invalid'));return;}
    const old=state;state=Object.assign({},state,{token:candidate,elapsed:0,duration:0,drafts:{},records:[]});busy=true;setDisabled(true);
    try{await resume();message('error','');}catch(e){state=old;message('error',t('invalid'));}finally{busy=false;setDisabled(false);}
  });
  $('copy-code').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(state.token);message('notice',t('copied'));}catch(e){message('notice',t('codeHelp'));}});
  $('backup').addEventListener('click',()=>{
    const blob=new Blob([JSON.stringify({study_id:B.study_id,participant_id:state.session.participant_id,ratings:state.records},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='my-ratings.json';a.target='_self';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.addEventListener('visibilitychange',()=>{last=Date.now();persist();});
  setInterval(()=>{
    const now=Date.now(),delta=Math.min(5000,now-last);last=now;
    if((view==='rating'||(view==='intro'&&!state.session))&&!busy&&!document.hidden){
      state.elapsed=Math.min(1800000,state.elapsed+delta);
      if(view==='rating'){state.duration+=delta;if(draft())draft().duration=state.duration;}persist();
      if(state.elapsed>=1800000){if(state.session){finish('time_limit');}else{setDisabled(false);message('notice',t('timeLimit'));}}}
  },1000);
  $('test-banner').classList.toggle('hidden',!B.is_test);translate();
  if(state.token&&state.session){busy=true;setDisabled(true);resume().catch(()=>{message('error',t('network'));}).finally(()=>{busy=false;setDisabled(false);});}
})();
