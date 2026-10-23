// Run the pinned upstream smoke with extra browser checkpoints; preserve every original assertion.
import { readFile, writeFile, mkdir, unlink } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { resolve, join, dirname } from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'

const [sourceArg, outArg] = process.argv.slice(2)
if (!sourceArg || !outArg) throw new Error('Usage: node verify-plugins.mjs SOURCE OUTPUT')
const source = resolve(sourceArg), out = resolve(outArg)
await mkdir(out, { recursive: true })
let code = await readFile(join(source, 'tools/public-docs/smoke.mjs'), 'utf8')
function insert(anchor, replacement) {
  if (!code.includes(anchor)) throw new Error(`Pinned smoke checkpoint not found: ${anchor}`)
  code = code.replace(anchor, replacement)
}
const captureUrl = pathToFileURL(join(dirname(fileURLToPath(import.meta.url)), 'capture.mjs')).href
insert("const repo = resolve(", `import { browser as eeBrowser, until as eeUntil } from ${JSON.stringify(captureUrl)}
let eePage
const eeOut = ${JSON.stringify(out)}
async function eeCapture(origin, session, name, expected) {
  eePage ??= await eeBrowser()
  await eePage.open(origin + '/?session=' + encodeURIComponent(session))
  if (expected) await eeUntil(() => eePage.evaluate('document.body.innerText.includes(' + JSON.stringify(expected) + ')'))
  await eePage.screenshot(join(eeOut, name + '.webp'))
}
const repo = resolve(`)
const installed = '    pass(`${name}: inspect/install/trust/enable`)'
insert(installed, installed + `
    if (name === 'panel') await eeCapture(origin, sessionId, '10-plugin-panel', 'Agnes client module demo')
`)
// Rollback is a profile snapshot operation: do not insert package changes between
// the original update and rollback assertions. Additional visual checks go after them.
const rolledBack = "  pass('rollback denies calls until explicit trust/enable restores v1')"
insert(rolledBack, rolledBack + `
  await op('disable', { id: '@agnes-examples/client-panel' })
  await eeCapture(origin, sessionId, '11-plugin-service-v1', 'backend 1.0.0')
  await op('update', { id: serviceId, source, expectedIntegrity: preview.integrity })
  await op('trust', { id: serviceId, expectedIntegrity: preview.integrity, capabilityHash: preview.capabilityHash })
  await op('enable', { id: serviceId })
  await eeCapture(origin, sessionId, '12-plugin-service-v2', 'backend 2.0.0')
  pass('browser panel v1 and linked backend v1/v2 rendered')`)
insert('} finally {', `} finally {
  await eePage?.close()`)
insert("  console.log(\n    `Evidence retained", `  await writeFile(join(eeOut, 'plugin-verification.json'), JSON.stringify({ stage, passed, ok: !process.exitCode, method: 'upstream loopback simulated model with browser checkpoints' }, null, 2))
  console.log(
    \`Evidence retained`)
const generated = join(source, 'tools/public-docs', `.every-embodied-${process.pid}.mjs`)
await writeFile(generated, code)
try {
  const child = spawn(process.execPath, ['--import', 'tsx', generated], { cwd: source, stdio: 'inherit' })
  const exitCode = await new Promise((done) => child.on('exit', done))
  process.exitCode = exitCode ?? 1
} finally { await unlink(generated) }
