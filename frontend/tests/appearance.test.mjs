import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const palettes=readFileSync(new URL('../public/themes.js',import.meta.url),'utf8');
const appearance=readFileSync(new URL('../public/appearance.js',import.meta.url),'utf8');
function boot(saved=null,dark=false,broken=false){
  const media={matches:dark,addEventListener:(_event,callback)=>media.changed=callback};
  let stored=saved;
  const style={setProperty(key,value){this[key]=value;},removeProperty(key){delete this[key];}};
  const context={window:{},document:{documentElement:{dataset:{},style}},matchMedia:()=>media,
    localStorage:{getItem:()=>{if(broken)throw Error('blocked');return stored;},setItem:(_key,value)=>{stored=value;}}};
  vm.createContext(context);vm.runInContext(palettes,context);vm.runInContext(appearance,context);
  return {api:context.window.fyodorAppearance,presets:context.window.fyodorThemes,root:context.document.documentElement,media,saved:()=>stored};
}
test('Customization validates persisted values, applies bounded styles and resets without losing sidebar state',()=>{
  const app=boot();
  app.api.set({accent:'#ffff00',codeColor:'#123456',fontSize:18,sidebarWidth:260,spacing:30,radius:0,font:'serif',mono:'courier',chatLayout:'wide',motion:'reduced',shadows:false,borders:false,translucency:true,collapsed:true});
  assert.equal(app.root.style['--on-accent'],'#000000');assert.equal(app.root.style['--ui-size'],'18px');
  assert.equal(app.root.style['--radius'],'0px');assert.equal(app.root.dataset.shadows,'false');
  const restored=boot(app.saved());assert.equal(restored.api.get().font,'serif');assert.equal(restored.root.style['--rail'],'260px');
  app.api.set({accent:'url(evil)',fontSize:999,sidebarWidth:-1,font:'<script>',radius:NaN});
  assert.equal(app.api.get().accent,'#ffff00');assert.equal(app.api.get().fontSize,18);assert.equal(app.api.get().font,'serif');
  app.api.set({accent:''});assert.equal(app.root.style['--accent'],undefined);
  app.api.reset();assert.equal(app.api.get().fontSize,14);assert.equal(app.api.get().collapsed,true);assert.equal(app.api.get().theme,'fyodor');
});
test('Required presets apply and persist; system tracks OS with explicit preferences preserved',()=>{
  const app=boot();assert.equal(app.api.get().theme,'fyodor');assert.equal(app.presets.length,10);
  for(const preset of app.presets){
    assert.equal(Object.keys(preset.roles).length,18);
    app.api.set({theme:preset.id});assert.equal(app.root.dataset.theme,preset.id);
    assert.equal(boot(app.saved()).api.get().theme,preset.id);
  }
  app.api.set({theme:'system',compact:true});app.media.matches=true;app.media.changed();
  assert.equal(app.root.dataset.theme,'fyodor-dark');assert.equal(app.api.get().theme,'system');
  app.media.matches=false;app.media.changed();assert.equal(app.root.dataset.theme,'fyodor');
  app.api.set({theme:'strawberry'});app.media.matches=true;app.media.changed();assert.equal(app.root.dataset.theme,'strawberry');
  app.api.set({theme:'unknown'});assert.equal(app.root.dataset.theme,'strawberry');
});
test('Existing theme preferences migrate and malformed/unavailable storage falls back safely',()=>{
  for(const [old,next] of Object.entries({'blue-black':'fyodor-dark','blue-white':'fyodor','orange-white':'fyodor-mocha',forest:'forest'})){
    const app=boot(JSON.stringify({theme:old,compact:true,collapsed:true}));assert.equal(app.api.get().theme,next);
    assert.equal(app.root.dataset.density,'compact');assert.equal(app.root.dataset.sidebar,'collapsed');
  }
  for(const saved of ['"text"','null','42','[]','bad'])assert.equal(boot(saved).api.get().theme,'fyodor');
  assert.equal(boot(null,false,true).api.get().theme,'fyodor');
});
test('All theme primary text pairs retain readable contrast',()=>{
  function luminance(hex){const rgb=hex.slice(1).match(/../g).map(v=>parseInt(v,16)/255).map(v=>v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4);return rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722;}
  const contrast=(a,b)=>{const x=luminance(a),y=luminance(b);return (Math.max(x,y)+.05)/(Math.min(x,y)+.05);};
  for(const {id,roles:r} of boot().presets){
    for(const bg of ['background','surface','surface_raised','selection','user_message','assistant_message'])
      assert.ok(contrast(r.text,r[bg])>=4.5,`${id} text/${bg}`);
    assert.ok(contrast(r.accent_text,r.accent)>=4.5,`${id} accent text`);
    assert.ok(contrast(r.accent_text,r.accent_hover)>=4.5,`${id} accent hover text`);
  }
});
