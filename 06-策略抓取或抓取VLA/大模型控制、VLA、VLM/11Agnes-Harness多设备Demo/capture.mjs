// Headless, isolated browser capture: webpage pixels only, never the desktop.
import { spawn } from 'node:child_process'
import { mkdtemp, readFile, mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
export async function until(fn, timeout = 60000) {
  const end = Date.now() + timeout
  while (Date.now() < end) {
    const value = await fn()
    if (value) return value
    await sleep(250)
  }
  throw new Error('Timed out waiting for verified state')
}

export async function browser() {
  const profile = await mkdtemp(join(tmpdir(), 'ee-agnes-browser-'))
  const child = spawn(process.env.CHROME || 'google-chrome', [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`,
    '--window-size=1600,900', '--hide-scrollbars', '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader', '--disable-background-networking',
    '--disable-component-update', '--no-first-run', 'about:blank',
    '--disable-renderer-backgrounding', '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
  ], { stdio: 'ignore' })
  const port = await until(async () => {
    if (child.exitCode !== null) throw new Error('Chrome exited before readiness')
    try { return Number((await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]) }
    catch { return false }
  })
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()
  const ws = new WebSocket(targets.find((target) => target.type === 'page').webSocketDebuggerUrl)
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true })
    ws.addEventListener('error', reject, { once: true })
  })
  let counter = 0
  const pending = new Map()
  const exceptions = []
  ws.addEventListener('message', ({ data }) => {
    const message = JSON.parse(data)
    if (message.method === 'Runtime.exceptionThrown')
      exceptions.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text)
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message)
      pending.delete(message.id)
    }
  })
  const send = async (method, params = {}) => {
    const id = ++counter
    const response = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)) }, 30000)
      pending.set(id, (value) => { clearTimeout(timer); resolve(value) })
      ws.send(JSON.stringify({ id, method, params }))
    })
    if (response.error) throw new Error(JSON.stringify(response.error))
    return response.result
  }
  await send('Runtime.enable')
  await send('Page.enable')
  // Headless Chrome may leave the initial target hidden; MHS subscribes only
  // while visible, and the Mars scene advances on requestAnimationFrame.
  await send('Emulation.setFocusEmulationEnabled', { enabled: true })
  await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 900, deviceScaleFactor: 1, mobile: false })
  const evaluate = async (expression) => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || 'Page evaluation failed')
    return result.result.value
  }
  return {
    send, evaluate, exceptions,
    async open(url, ready = 'document.readyState === "complete"') {
      const result = await send('Page.navigate', { url })
      if (result.errorText) throw new Error(result.errorText)
      await send('Page.bringToFront')
      await send('Emulation.setFocusEmulationEnabled', { enabled: true })
      await until(async () => { try { return await evaluate(ready) } catch { return false } }, 180000)
      await sleep(1200)
    },
    async screenshot(path) {
      await mkdir(join(path, '..'), { recursive: true })
      const shot = await send('Page.captureScreenshot', { format: 'webp', quality: 90 })
      await writeFile(path, Buffer.from(shot.data, 'base64'))
      console.log(`SCREENSHOT ${path}`)
    },
    async key(key, code = key) {
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key, code })
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key, code })
      await sleep(500)
    },
    async close() {
      ws.close()
      child.kill('SIGTERM')
    },
  }
}
