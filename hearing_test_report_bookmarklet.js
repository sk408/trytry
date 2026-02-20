javascript:(async function(){
  let C=function(s){return document.createElement(s)};

  const fetchOpts = { credentials: "include" };

  async function e(url,retries){
    retries=retries===undefined?2:retries;
    for(let attempt=0;attempt<=retries;attempt++){
      try{
        let t=await fetch("https://costco.sycle.net/api"+url,fetchOpts);
        if(!t.ok){
          let retryable=t.status>=500||t.status===429;
          if(attempt<retries&&retryable){
            await new Promise(function(r){setTimeout(r,1000*(attempt+1))});
            continue
          }
          let err=new Error("Fetch failed: "+url+" ("+t.status+")");
          err.noRetry=true;
          throw err
        }
        return await t.json()
      }catch(err){
        if(attempt<retries&&!err.noRetry&&err.name!=="AbortError"){
          await new Promise(function(r){setTimeout(r,1000*(attempt+1))});
          continue
        }
        throw err
      }
    }
  }

  async function t(id){
    let now=new Date();
    let backto="/schedule/?day="+now.getDate()+"&month="+(now.getMonth()+1)+"&year="+now.getFullYear()+"&mode=custom_day";
    await fetch("https://costco.sycle.net/clinic/switch/"+id+"/?backto="+encodeURIComponent(backto),fetchOpts)
  }

  async function a(){
    let t=await e("/scheduler/staff?func[editable]=1&order_by[staffLast]=asc&order_by[staffFirst]=asc");
    return t.data.filter(function(x){return x.type==="Provider"}).map(function(x){return{name:x.firstName+" "+x.lastName,id:x.id}})
  }

  async function n(){
    let t=await e("/clinic-management/clinics");
    let a=[];
    t.data.forEach(function(x){if(x.clinicName)a.push({id:x.id,name:x.clinicName})});
    return a
  }

  function l(e){
    let t=e.match(/#(\d+)/);
    return t?parseInt(t[1]):999999
  }

  async function i(yr,mo,dy){
    return await e("/calendars?year="+yr+"&month="+mo+"&day="+dy)
  }

  async function c(t){
    let a=await e("/appointment/summary/main/data?appt_id="+t);
    return{outcomeNotes:a.data.outcomeNotes||"",appointmentType:a.data.appointmentType||""}
  }

  async function fetchInvoices(pid){
    let n=await e("/finance/invoices?filter[patientId][eq]="+pid);
    return n.data||[]
  }

  function hasPurchaseAfter(invoices,dateStr){
    if(!invoices||invoices.length===0)return false;
    let d=parseLocalDate(dateStr);
    return invoices.some(function(x){return parseLocalDate(x.purchaseDate)>=d})
  }

  function daysSinceMonday(d){
    let a=d.getDay();
    return a===1?0:a===0?6:a-1
  }

  function parseLocalDate(str){
    if(!str)return new Date();
    let p=str.split(/[-T]/);
    if(p.length>=3)return new Date(+p[0],+p[1]-1,+p[2]);
    return new Date(str)
  }

  function daysBetween(d1,d2){
    let a=Date.UTC(d1.getFullYear(),d1.getMonth(),d1.getDate());
    let b=Date.UTC(d2.getFullYear(),d2.getMonth(),d2.getDate());
    return Math.round((b-a)/864e5)
  }

  function d(e,t){
    let a=new Date(2026,1,17);
    let n=daysSinceMonday(a);
    let i=new Date(a);
    i.setDate(i.getDate()-n);
    let c=(e-7)*28;
    let r=new Date(i);
    r.setDate(r.getDate()+c);
    let s=new Date(r);
    let o=new Date(r);
    if(t==="all"||t===0){
      o.setDate(o.getDate()+27)
    }else{
      let p=parseInt(t);
      s.setDate(s.getDate()+(p-1)*7);
      o=new Date(s);
      o.setDate(o.getDate()+6)
    }
    return{start:s,end:o}
  }

  function s(e){
    let t=new Date(2026,1,17);
    let n=daysSinceMonday(t);
    let l=new Date(t);
    l.setDate(l.getDate()-n);
    let i=parseLocalDate(e);
    let c=daysBetween(l,i);
    let r=Math.floor(c/28)+7;
    let d=Math.floor((c%28)/7)+1;
    let s=(c%7)+1;
    if(c<0){
      let o=Math.abs(c);
      r=7-Math.ceil(o/28);
      let dayInPeriod=28-((o-1)%28);
      d=Math.ceil(dayInPeriod/7);
      s=((dayInPeriod-1)%7)+1
    }
    return"P"+r+"W"+d+"D"+s
  }

  function H(x){
    if(!x)return"";
    let el=C("div");
    el.innerHTML=x;
    return(el.textContent||"").replace(/\s+/g," ").trim()
  }

  function csvEscape(str){
    if(!str)return"";
    return str.replace(/"/g,'""').replace(/\r?\n/g," ").replace(/\s+/g," ").trim()
  }

  function outcomeToCsv(str){
    if(!str)return"";
    let el=C("div");
    el.innerHTML=str;
    return(el.textContent||"").replace(/"/g,'""').replace(/\s+/g," ").trim().substring(0,500)
  }

  function batchProcess(items,fn,size,onProgress){
    size=size||4;
    var idx=0;
    return{
      async run(){
        var results=[];
        while(idx<items.length){
          var batch=[];
          var batchItems=[];
          for(var x=idx;x<idx+size&&x<items.length;x++){
            batchItems.push(items[x]);
            batch.push(fn(items[x]))
          }
          var rs=await Promise.all(batch);
          for(var x=0;x<rs.length;x++)results.push({key:batchItems[x],val:rs[x]});
          idx+=size;
          if(onProgress)onProgress(Math.min(idx,items.length),items.length)
        }
        return results
      }
    }
  }

  async function p(clinics,staffFilter,st,days,n){
    let staffMap={};
    let startDate=parseLocalDate(st);
    let endDate=new Date(startDate);
    endDate.setDate(endDate.getDate()+days);

    for(let cid of clinics){
      n.textContent="Accessing Clinic "+cid+"...";
      await t(cid);

      let chunkStart=new Date(startDate);
      let remaining=days;

      while(remaining>0){
        let yr=chunkStart.getFullYear();
        let mo=chunkStart.getMonth()+1;
        let dy=chunkStart.getDate()+1;

        let cal=await i(yr,mo,dy);

        if(cal.data){
          cal.data.forEach(day=>{
            let dayDate=parseLocalDate(day.date);
            if(dayDate>=startDate&&dayDate<endDate){
              if(day.appointments){
                day.appointments.forEach(appt=>{
                  let sName=appt.staffFullName;
                  if(String(appt.clinicId)===String(cid)&&
                    (staffFilter.length===0||staffFilter.includes(sName))){
                    if(!staffMap[sName])staffMap[sName]={};
                    if(appt.apptName){
                      staffMap[sName][appt.patientId]={
                        date:appt.startTime.split(" ")[0],
                        apptId:appt.apptId,
                        message:appt.notes&&appt.notes.length>0?appt.notes[0].message:""
                      }
                    }
                  }
                })
              }
            }
          })
        }

        chunkStart.setDate(chunkStart.getDate()+28);
        remaining-=28;
      }
    }

    let staffNames=Object.keys(staffMap);

    // Collect all unique patients across all staff
    let uniquePatients={};
    staffNames.forEach(function(sName){
      Object.keys(staffMap[sName]).forEach(function(pid){
        if(!uniquePatients[pid])uniquePatients[pid]=[];
        let ap=staffMap[sName][pid];
        uniquePatients[pid].push({staffName:sName,date:ap.date,apptId:ap.apptId,message:ap.message})
      })
    });
    let allPids=Object.keys(uniquePatients);

    // Phase 1: Patient-stats — one call per unique patient
    n.textContent="Checking patients... 0/"+allPids.length;
    let statsCache={};
    let statsResults=await batchProcess(allPids,function(pid){
      return e("/patient-stats/"+pid).catch(function(){return{data:{}}})
    },4,function(done,total){
      n.textContent="Checking patients... "+done+"/"+total
    }).run();
    statsResults.forEach(function(r){statsCache[r.key]=r.val.data});

    // Filter confirmed tests
    let confirmedTests=[];
    let testCountByStaff={};
    staffNames.forEach(function(sName){testCountByStaff[sName]=0});

    allPids.forEach(function(pid){
      let stats=statsCache[pid];
      if(!stats||!stats.lastHearingTest)return;
      uniquePatients[pid].forEach(function(ap){
        if(stats.lastHearingTest.date===ap.date){
          testCountByStaff[ap.staffName]=(testCountByStaff[ap.staffName]||0)+1;
          confirmedTests.push({patientId:pid,apptId:ap.apptId,message:ap.message,apptDate:ap.date,staffName:ap.staffName})
        }
      })
    });

    // Phase 2: Appointment summaries — deduplicated by apptId
    let seenAppt={};
    let uniqueApptIds=[];
    confirmedTests.forEach(function(t){
      if(!seenAppt[t.apptId]){seenAppt[t.apptId]=true;uniqueApptIds.push(t.apptId)}
    });
    n.textContent="Details... 0/"+uniqueApptIds.length;
    let summaryCache={};
    let sumResults=await batchProcess(uniqueApptIds,function(aid){
      return c(aid).catch(function(){return{outcomeNotes:"",appointmentType:""}})
    },4,function(done,total){
      n.textContent="Details... "+done+"/"+total
    }).run();
    sumResults.forEach(function(r){summaryCache[r.key]=r.val});

    // Phase 3: Invoices — deduplicated by patientId
    let seenInv={};
    let uniqueInvPids=[];
    confirmedTests.forEach(function(t){
      if(!seenInv[t.patientId]){seenInv[t.patientId]=true;uniqueInvPids.push(t.patientId)}
    });
    n.textContent="Purchases... 0/"+uniqueInvPids.length;
    let invoiceCache={};
    let invResults=await batchProcess(uniqueInvPids,function(pid){
      return fetchInvoices(pid).catch(function(){return[]})
    },4,function(done,total){
      n.textContent="Purchases... "+done+"/"+total
    }).run();
    invResults.forEach(function(r){invoiceCache[r.key]=r.val});

    // Assemble results per staff
    let testsByStaff={};
    confirmedTests.forEach(function(t){
      if(!testsByStaff[t.staffName])testsByStaff[t.staffName]=[];
      let sum=summaryCache[t.apptId]||{outcomeNotes:"",appointmentType:""};
      let inv=invoiceCache[t.patientId]||[];
      testsByStaff[t.staffName].push({
        patientId:t.patientId,apptId:t.apptId,message:t.message,apptDate:t.apptDate,
        outcomeNotes:sum.outcomeNotes,appointmentType:sum.appointmentType,
        hasPurchase:hasPurchaseAfter(inv,t.apptDate)
      })
    });

    // Build UI results
    let finalRes=[];
    staffNames.forEach(function(sName){
      let tests=testsByStaff[sName]||[];
      tests.sort(function(a,b){return parseLocalDate(a.apptDate)-parseLocalDate(b.apptDate)});
      let count=testCountByStaff[sName]||0;
      let f=tests.filter(function(t){return t.hasPurchase}).length;
      let m=(count>0?f/count*100:0).toFixed(1);
      let u=C("div");
      u.style.marginBottom="15px";
      u.style.borderBottom="1px solid #ccc";
      u.style.paddingBottom="10px";
      let h=C("strong");
      h.textContent=sName+": "+count+" tests, "+f+" ("+m+"%)";
      u.appendChild(h);
      if(tests.length>0){
        let y=C("ul");
        y.style.listStyle="none";
        y.style.padding="5px";
        tests.forEach(function(t){
          let li=C("li");
          li.style.marginBottom="8px";
          li.style.padding="8px";
          li.style.backgroundColor=t.hasPurchase?"#dfd":"#fdd";
          let a=C("a");
          a.href="https://costco.sycle.net/appointment/outcome?appt_id="+t.apptId;
          a.textContent="ID:"+t.patientId;
          a.target="_blank";
          li.appendChild(a);
          let sp=C("span");
          sp.textContent=" "+t.apptDate+" "+(t.appointmentType||"");
          li.appendChild(sp);
          if(t.message){
            let dv=C("div");
            dv.style.fontSize="12px";
            dv.textContent="Notes: "+H(t.message);
            li.appendChild(dv)
          }
          if(t.outcomeNotes){
            let dv2=C("div");
            dv2.style.fontSize="12px";
            dv2.textContent="Outcome: "+t.outcomeNotes.replace(/<[^>]*>/g,"").replace(/&nbsp;/g," ");
            li.appendChild(dv2)
          }
          y.appendChild(li)
        });
        u.appendChild(y)
      }
      finalRes.push({element:u,data:{staff:sName,tests:tests,totalTests:count,purchases:f,captureRate:m}})
    });
    return finalRes
  }

  function rowToCsv(test,staffData,isFirst){
    let msg=csvEscape(test.message?H(test.message):"");
    let out=outcomeToCsv(test.outcomeNotes);
    let typ=test.appointmentType||"";
    let lnk="https://costco.sycle.net/appointment/outcome?appt_id="+test.apptId;
    let pwd=s(test.apptDate);
    if(isFirst){
      return '"'+staffData.staff+'","'+staffData.totalTests+'","'+staffData.purchases+'","'+staffData.captureRate+'%","'+test.patientId+'","'+typ+'","'+test.apptDate+'","'+pwd+'","'+msg+'","'+out+'","'+(test.hasPurchase?"Yes":"No")+'","'+lnk+'"';
    }
    return '"","","","","'+test.patientId+'","'+typ+'","'+test.apptDate+'","'+pwd+'","'+msg+'","'+out+'","'+(test.hasPurchase?"Yes":"No")+'","'+lnk+'"';
  }

  let cl,st;
  try{
    cl=await n();
    st=await a();
  }catch(err){
    alert("Failed to load data: "+err.message+"\n\nMake sure you are logged in on costco.sycle.net.");
    return;
  }

  let u=[];
  let activeMode="per";
  let savedBody=document.body.cloneNode(true);

  (function(){
    document.body.innerHTML="";
    let i=C("div");
    i.style.cssText="max-width:900px;margin:20px auto;padding:20px;border:1px solid #ccc";
    let hdr=C("h1");
    hdr.textContent="Hearing Test Report";
    i.appendChild(hdr);
    let closeBtn=C("button");
    closeBtn.textContent="Close (Restore Page)";
    closeBtn.style.cssText="float:right;padding:8px 12px;margin-top:-10px";
    closeBtn.onclick=function(){
      document.body.innerHTML="";
      document.body.appendChild(savedBody);
    };
    hdr.appendChild(closeBtn);
    let r=C("div");
    r.style.marginBottom="20px";
    let lbl=C("label");
    lbl.textContent="Select Clinic(s) (Required):";
    r.appendChild(lbl);
    let y=C("select");
    y.multiple=true;
    y.size=10;
    y.style.cssText="width:100%;height:200px;margin-bottom:10px";
    cl.sort(function(a,b){return l(a.name)-l(b.name)});
    cl.forEach(function(x){
      let opt=C("option");
      opt.value=x.id;
      opt.textContent=x.name;
      y.appendChild(opt)
    });
    r.appendChild(y);
    let lbl2=C("label");
    lbl2.textContent="Filter by Staff (Optional - Leave blank for all):";
    r.appendChild(lbl2);
    let v=C("select");
    v.multiple=true;
    v.size=10;
    v.style.cssText="width:100%;height:200px";
    st.forEach(function(x){
      let opt=C("option");
      opt.value=x.name;
      opt.textContent=x.name;
      v.appendChild(opt)
    });
    r.appendChild(v);
    i.appendChild(r);
    let w=C("div");
    w.style.marginBottom="20px";
    let k=C("div");
    k.style.cssText="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px";
    function createBtn(txt,mode){
      let btn=C("button");
      btn.textContent=txt;
      btn.style.padding="15px";
      btn.dataset.mode=mode;
      btn.onclick=function(){
        document.querySelectorAll("[data-mode]").forEach(function(b){b.style.background=""});
        btn.style.background="#ccf";
        activeMode=mode;
        document.querySelectorAll(".opts").forEach(function(o){o.style.display="none"});
        document.getElementById(mode+"Opts").style.display="block"
      };
      return btn
    }
    let D=createBtn("Date Range","cal");
    let x=createBtn("Single Period","per");
    let I=createBtn("Multi Period","bat");
    k.appendChild(D);
    k.appendChild(x);
    k.appendChild(I);
    w.appendChild(k);
    let P=C("div");
    P.className="opts";
    P.id="calOpts";
    P.style.cssText="display:none;margin-top:10px";
    let S=C("input");
    S.type="date";
    S.id="startDate";
    S.style.marginRight="10px";
    let E=C("input");
    E.type="date";
    E.id="endDate";
    let defStart=new Date();
    defStart.setDate(defStart.getDate()-30);
    let defEnd=new Date();
    S.value=defStart.toISOString().split("T")[0];
    E.value=defEnd.toISOString().split("T")[0];
    P.appendChild(document.createTextNode("Start: "));
    P.appendChild(S);
    P.appendChild(document.createTextNode(" End: "));
    P.appendChild(E);
    w.appendChild(P);
    let T=C("div");
    T.className="opts";
    T.id="perOpts";
    T.style.cssText="display:block;margin-top:10px";
    x.style.background="#ccf";
    let M=C("select");
    M.id="periodSelect";
    M.style.marginRight="10px";
    for(let N=1;N<=13;N++){
      let opt=C("option");
      opt.value=N;
      opt.textContent="Period "+N;
      if(N===7){opt.selected=true}
      M.appendChild(opt)
    }
    let A=C("select");
    A.id="weekSelect";
    let L=C("option");
    L.value="all";
    L.textContent="Full Period";
    A.appendChild(L);
    for(let $=1;$<=4;$++){
      let opt=C("option");
      opt.value=$;
      opt.textContent="Week "+$;
      A.appendChild(opt)
    }
    T.appendChild(document.createTextNode("Period: "));
    T.appendChild(M);
    T.appendChild(document.createTextNode(" Week: "));
    T.appendChild(A);
    w.appendChild(T);
    let R=C("div");
    R.className="opts";
    R.id="batOpts";
    R.style.cssText="display:none;margin-top:10px";
    let _=C("select");
    _.id="batchSelect";
    _.multiple=true;
    _.size=13;
    _.style.cssText="width:200px;height:260px";
    for(let j=1;j<=13;j++){
      let opt=C("option");
      opt.value=j;
      opt.textContent="Period "+j;
      _.appendChild(opt)
    }
    let z=C("select");
    z.id="batchWeek";
    z.style.marginLeft="10px";
    let B=C("option");
    B.value="all";
    B.textContent="Full Period";
    z.appendChild(B);
    for(let F=1;F<=4;F++){
      let opt=C("option");
      opt.value=F;
      opt.textContent="Week "+F;
      z.appendChild(opt)
    }
    R.appendChild(document.createTextNode("Periods: "));
    R.appendChild(_);
    R.appendChild(z);
    w.appendChild(R);
    i.appendChild(w);
    let J=C("div");
    J.id="progress";
    J.style.cssText="display:none;padding:10px;background:#eef;margin-bottom:10px";
    i.appendChild(J);
    let K=C("button");
    K.textContent="Generate";
    K.style.cssText="width:100%;padding:15px;font-size:16px";
    K.onclick=async function(){
      let res=document.getElementById("results");
      if(res){res.remove()}
      let btn=document.getElementById("downloadBtn");
      if(btn){btn.remove()}
      let cIds=Array.from(y.options).filter(function(o){return o.selected}).map(function(o){return o.value});
      if(cIds.length===0){alert("Select at least one clinic");return}
      let sNames=Array.from(v.options).filter(function(o){return o.selected}).map(function(o){return o.value});
      J.style.display="block";
      J.textContent="Processing...";
      K.disabled=true;
      u=[];
      try{
        if(activeMode==="bat"){
          let pers=Array.from(_.options).filter(function(o){return o.selected}).map(function(o){return parseInt(o.value)});
          if(pers.length===0){J.style.display="none";K.disabled=false;alert("Select periods");return}
          let wk=z.value;
          for(let c=0;c<pers.length;c++){
            let per=pers[c];
            J.textContent="Period "+per+" ("+(c+1)+"/"+pers.length+")";
            let rng=d(per,wk);
            let st=rng.start;
            let en=rng.end;
            let days=daysBetween(st,en)+1;
            let results=await p(cIds,sNames,st.toISOString().split("T")[0],days,J);
            results.forEach(function(r){u.push({period:per,data:r.data})})
          }
          downloadBatch(u);
          J.style.display="none";
          K.disabled=false;
          alert("Downloaded "+pers.length+" files")
        }else{
          let st,en,days;
          if(activeMode==="cal"){
            st=parseLocalDate(S.value);
            en=parseLocalDate(E.value);
            days=daysBetween(st,en)+1;
            if(st.toString()==="Invalid Date"||en.toString()==="Invalid Date"){
              J.style.display="none";K.disabled=false;alert("Invalid dates");return
            }
          }else{
            let per=parseInt(M.value);
            let wk=A.value;
            let rng=d(per,wk);
            st=rng.start;
            en=rng.end;
            days=daysBetween(st,en)+1
          }
          S.value=st.toISOString().split("T")[0];
          E.value=en.toISOString().split("T")[0];
          let cont=C("div");
          cont.id="results";
          cont.style.marginTop="20px";
          J.textContent="Processing...";
          let results=await p(cIds,sNames,S.value,days,J);
          results.forEach(function(r){cont.appendChild(r.element);u.push({data:r.data})});
          i.appendChild(cont);
          J.style.display="none";
          K.disabled=false;
          let btn2=C("button");
          btn2.id="downloadBtn";
          btn2.textContent="Download CSV";
          btn2.style.cssText="width:100%;padding:12px;margin-top:10px";
          btn2.onclick=function(){downloadCSV(u)};
          cont.parentNode.insertBefore(btn2,cont)
        }
      }catch(err){
        J.style.display="none";
        K.disabled=false;
        J.textContent="";
        alert("Error: "+err.message);
      }
    };
    i.appendChild(K);
    document.body.appendChild(i)
  })();

  function downloadCSV(data){
    let csv="Staff,Total Tests,Purchases,Capture Rate,Patient ID,Appointment Type,Appointment Date,Period Week Day,Notes,Outcome Notes,Has Purchase,Appointment Link\n";
    data.forEach(function(item){
      item.data.tests.forEach(function(test,idx){
        csv+=rowToCsv(test,item.data,idx===0)+"\n";
      })
    });
    let blob=new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"});
    let url=URL.createObjectURL(blob);
    let a=C("a");
    a.href=url;
    a.download="hearing_tests.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadBatch(data){
    let grouped={};
    data.forEach(function(item){
      if(!grouped[item.period]){grouped[item.period]=[]}
      grouped[item.period].push(item.data)
    });
    Object.keys(grouped).forEach(function(per){
      let csv="Staff,Total Tests,Purchases,Capture Rate,Patient ID,Appointment Type,Appointment Date,Period Week Day,Notes,Outcome Notes,Has Purchase,Appointment Link\n";
      grouped[per].forEach(function(staffData){
        staffData.tests.forEach(function(test,idx){
          csv+=rowToCsv(test,staffData,idx===0)+"\n";
        })
      });
      let blob=new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"});
      let url=URL.createObjectURL(blob);
      let a=C("a");
      a.href=url;
      a.download="Period_"+per+"_tests.csv";
      a.click();
      URL.revokeObjectURL(url);
    })
  }

})();
