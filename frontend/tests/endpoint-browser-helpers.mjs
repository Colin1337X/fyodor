// Shared public Settings interaction; synthetic credentials exist only in the
// owned loopback fixture/private profile, never a real vendor account.
export async function configureAgentFixture({ evaluate, waitFor }) {
  const endpoint = process.env.FYODOR_AGENT_ENDPOINT;
  await evaluate('document.querySelector("#open-settings").click()');
  await evaluate(
    `(()=>{const d=document.querySelector('#settings');d.querySelector('#provider-type').value='openai';d.querySelector('#provider-type').dispatchEvent(new Event('input',{bubbles:true}));d.querySelector('#provider-endpoint').value=${JSON.stringify(endpoint)};d.querySelector('#provider-model').value='fixture-agent-model';d.querySelector('#provider-key').value='agent-fixture-key';d.querySelector('#remember-key').checked=false;d.querySelector('#agent-enabled').checked=false;d.querySelector('#save-settings').click();})()`,
  );
  await waitFor('!document.querySelector("#settings").open');
}
