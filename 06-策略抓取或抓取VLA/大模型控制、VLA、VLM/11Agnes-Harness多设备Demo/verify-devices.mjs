// Executes real Hub calls against virtual devices; no model or scripted page-state changes.
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { open, mkdir, writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { browser, until, sleep } from './capture.mjs'

const [sourceArg, outputArg, mode = 'all'] = process.argv.slice(2)
if (!sourceArg || !outputArg) throw new Error('Usage: node verify-devices.mjs SOURCE OUTPUT [office|mars|all]')
const source = resolve(sourceArg), out = resolve(outputArg)
await mkdir(out, { recursive: true })
const processes = [], checks = [], calls = []
let page, hub
async function start(name, args) {
  const log = await open(join(out, `${name}.log`), 'w')
  const child = spawn('pnpm', args, { cwd: source, detached: true,
    stdio: ['ignore', log.fd, log.fd] })
  processes.push(child)
  await log.close()
  return child
}
async function connect(port) {
  // The command-line Hub intentionally serves HTTP 404 at /; its WebSocket is the API.
  await until(async () => { try { return (await fetch(`http://127.0.0.1:${port}/`)).status > 0 } catch { return false } })
  const ws = new WebSocket(`ws://127.0.0.1:${port}/ws/hub`)
  await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }) })
  let next = 0
  const pending = new Map(), results = new Map()
  ws.addEventListener('message', ({ data }) => {
    if (typeof data !== 'string') return
    const message = JSON.parse(data)
    if (message.method === 'hub/result') results.set(message.params.job, message.params)
    if (message.id && pending.has(message.id)) { pending.get(message.id)(message); pending.delete(message.id) }
  })
  const req = async (method, params = {}) => {
    const id = String(++next)
    const message = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Hub timeout ${method}`)) }, 15000)
      pending.set(id, (value) => { clearTimeout(timer); resolve(value) })
      ws.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }))
    })
    if (message.error) throw new Error(JSON.stringify(message.error))
    return message.result
  }
  await req('hub/hello', { role: 'ui', client: { name: 'every-embodied-verifier' } })
  return {
    req, close: () => ws.close(),
    async device(id) { return (await req('hub/devices')).devices.find((device) => device.id === id) },
    async call(device, tool, args = {}, timeout = 420000) {
      console.log(`CALL ${device}.${tool} ${JSON.stringify(args)}`)
      const started = Date.now()
      const accepted = await req('hub/call', { device, tool, arguments: args })
      assert.equal(accepted.accepted, true, JSON.stringify(accepted))
      const result = await until(() => results.get(accepted.job), timeout)
      calls.push({ device, tool, arguments: args, result, elapsed_ms: Date.now() - started })
      await writeFile(join(out, 'calls.json'), JSON.stringify(calls, null, 2))
      assert.equal(result.status, 'done', JSON.stringify(result))
      console.log(`DONE ${device}.${tool}: ${result.detail}`)
      return result
    },
  }
}
function pass(name, evidence) { checks.push({ name, pass: true, evidence }); console.log(`PASS ${name}`) }
async function photo(name) { await page.screenshot(join(out, `${name}.webp`)) }
async function reset() { await page.key('r', 'KeyR'); await sleep(1000) }

try {
  page = await browser()
  if (mode !== 'mars') {
    await start('office', ['--filter', '@agnes/mhs', 'dev-hub'])
    hub = await connect(4191)
    await until(async () => (await hub.req('hub/devices')).devices.filter((d) => d.available).length === 4)
    await page.open('http://127.0.0.1:4191/?lang=zh-CN', 'document.body.innerText.includes("robot-01")')
    await page.evaluate('[...document.querySelectorAll("button")].find(b => b.textContent.trim() === "设备").click()')
    await sleep(1000)
    await photo('01-office-devices')
    await hub.req('hub/set', { device: 'lamp-01', values: { on: true, brightness: 30, color: 'cool' } })
    const lamp = (await hub.device('lamp-01')).state.values
    assert.equal(lamp.on, true); assert.equal(lamp.brightness, 30); assert.equal(lamp.color, 'cool')
    await page.open('http://127.0.0.1:4191/?lang=zh-CN&device=lamp-01', 'document.body.innerText.includes("lamp-01")')
    await photo('02-office-lamp')
    pass('office writable lamp state', lamp)
    await hub.call('robot-01', 'drive_to', { target: { x: 5, y: 4 } })
    const robot = await hub.device('robot-01')
    assert.ok(Math.hypot(robot.position.x - 5, robot.position.y - 4) < 0.7)
    const reading = await hub.req('hub/read', { device: 'robot-01', sources: ['front'] })
    assert.ok(reading.items.some((i) => i.kind === 'image' && i.b64?.length > 100))
    await page.open('http://127.0.0.1:4191/?lang=zh-CN&device=robot-01', 'document.body.innerText.includes("robot-01")')
    await until(() => page.evaluate('[...document.querySelectorAll("canvas")].some(c => c.width === 160 && c.height === 90)'))
    await sleep(1000); await photo('03-office-robot')
    pass('office motion and camera read', { position: robot.position, image_received: true })
    hub.close(); hub = undefined
  }
  if (mode !== 'office') {
    await start('mars-hub', ['--filter', '@agnes/mhs', 'dev', '127.0.0.1:4180'])
    hub = await connect(4180)
    await start('mars-world', ['--filter', '@agnes/mars-world', 'dev'])
    await until(async () => { try { return (await fetch('http://127.0.0.1:4200/')).ok } catch { return false } })
    await page.open('http://127.0.0.1:4200/?hub=ws://127.0.0.1:4180&quality=low', 'document.body.dataset.ready === "1"')
    await until(async () => (await hub.req('hub/devices')).devices.filter((d) => d.available).length === 11)
    await photo('04-mars-overview')
    pass('Mars 11 devices connected', (await hub.req('hub/devices')).devices.map((d) => d.id))

    // Rescue uses live suit position, real motion and the airlock's own interlocks.
    await hub.call('hopper-01', 'take_off', { alt: 10 })
    const rescueSurvey = await hub.call('hopper-01', 'survey', { area: 'west field' })
    assert.ok(rescueSurvey.data.found.some((thing) => thing.id === 'suit-01'))
    await hub.call('hopper-01', 'land', { pad: 'hopper pad' })
    await hub.call('monolith-01', 'follow', { target: 'suit-01' })
    await hub.call('monolith-01', 'carry', { target: 'suit-01' })
    assert.equal((await hub.device('monolith-01')).state.values.carrying, 'astronaut')
    await page.key('0', 'Digit0'); await photo('05-mars-rescue-carry')
    await hub.call('airlock-01', 'cycle_out')
    await hub.call('airlock-01', 'open_outer')
    await hub.call('monolith-01', 'go_to', { x: 0, y: -12.2, gait: 'walk' })
    await hub.call('monolith-01', 'put_down')
    assert.equal((await hub.device('airlock-01')).state.values.occupied, true)
    await page.key('4', 'Digit4'); await photo('06-mars-airlock')
    await hub.call('airlock-01', 'close_outer')
    const rescued = await hub.call('airlock-01', 'cycle_in')
    assert.match(rescued.detail, /stepped into the habitat/)
    pass('Mars rescue through pressurized airlock', rescued.detail)

    await reset()
    await hub.call('hopper-01', 'take_off', { alt: 10 })
    const survey = await hub.call('hopper-01', 'survey', { area: 'west field' })
    const rock = survey.data.found.find((item) => item.label === 'unusual rock')
    assert.ok(rock); assert.ok(survey.data.route?.length)
    await hub.call('hopper-01', 'land', { pad: 'hopper pad' })
    await hub.call('rover-01', 'follow_route', { points: survey.data.route })
    await hub.call('rover-01', 'collect_sample', { target: rock.id })
    await hub.call('rover-01', 'dock')
    const unloaded = await hub.call('lab-01', 'unload_rover')
    assert.ok(unloaded.data.samples.includes(rock.id))
    const finding = await hub.call('lab-01', 'analyze', { sample: rock.id })
    assert.match((await hub.device('lab-01')).state.values.last_result, new RegExp(rock.id))
    await page.key('2', 'Digit2'); await photo('08-mars-lab')
    await hub.call('comms-01', 'point', { target: 'earth' })
    await until(async () => (await hub.device('comms-01')).state.values.window === 'open', 250000)
    const sent = await hub.call('comms-01', 'send', { report: finding.detail })
    pass('Mars survey/sample/lab/Earth report', { sample: rock.id, finding: finding.data, sent: sent.detail })

    await reset()
    await hub.call('power-01', 'stow_array')
    await hub.call('rover-01', 'dock')
    await hub.call('habitat-01', 'set_mode', { mode: 'storm_shelter' })
    const power = (await hub.device('power-01')).state.values
    const habitat = (await hub.device('habitat-01')).state.values
    assert.equal(power.panels, 'stowed'); assert.equal(habitat.mode, 'storm_shelter')
    assert.equal((await hub.device('hopper-01')).state.values.mode, 'landed')
    await page.key('6', 'Digit6'); await photo('09-mars-storm-power')
    pass('Mars storm preparation actions', { power, habitat, hopper: 'landed' })
  }
  assert.equal(page.exceptions.length, 0, page.exceptions.join('\n'))
} catch (error) {
  console.error(error.stack)
  try {
    console.error('PAGE', await page.evaluate('JSON.stringify({visibility:document.visibilityState,ready:document.body.dataset.ready,text:document.body.innerText.slice(0,1500)})'))
    await page.screenshot(join(out, 'failure.webp'))
  } catch {}
  checks.push({ name: 'execution error', pass: false, error: error.message })
  process.exitCode = 1
} finally {
  await writeFile(join(out, 'device-verification.json'), JSON.stringify({ mode,
    method: 'scripted Hub API calls, real virtual-device execution, no language model',
    checks, ok: !process.exitCode, page_exceptions: page?.exceptions || [] }, null, 2))
  hub?.close(); await page?.close()
  for (const child of processes) { try { process.kill(-child.pid, 'SIGTERM') } catch {} }
}
