const vscode = require('vscode');
const { execFile } = require('node:child_process');
const { createHash, randomUUID } = require('node:crypto');
const localFs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { promisify } = require('node:util');

const requestMime = 'application/vnd.pgl.dialog+json';
const requestLimit = 64 * 1024;
const desktopPythonKey = 'desktopPython';
const run = promisify(execFile);
const dialogs = new Map([
    ['confirm', { validate: validConfirmation, script: 'confirm.py', resultKey: 'confirmed' }],
    ['selectItems', { validate: validSelection, script: 'selectItems.py', resultKey: 'result' }],
]);

// =============================================================================
// Receive notebook requests and send replies over VS Code's remote filesystem
// =============================================================================

// =============================================================================
async function serveRequest(root, id, desktopPython) {
    const fs = vscode.workspace.fs;
    const requestFile = vscode.Uri.joinPath(root, 'request.json');
    const stateFile = vscode.Uri.joinPath(root, 'response.json');
    const pendingFile = vscode.Uri.joinPath(root, `response-${randomUUID()}.tmp`);

    const directory = await fs.stat(root);
    const file = await fs.stat(requestFile);
    if (directory.type !== vscode.FileType.Directory || file.type !== vscode.FileType.File || file.size > requestLimit) return;
    const data = await fs.readFile(requestFile);
    if (data.length > requestLimit) return;
    const { version, mode, request: requestText } = JSON.parse(Buffer.from(data).toString('utf8'));
    const dialog = dialogs.get(mode);
    if (![1, 2].includes(version) || !dialog || typeof requestText !== 'string') return;
    const request = JSON.parse(requestText);
    if (!dialog.validate(request)) return;
    const requestHash = createHash('sha256').update(requestText, 'utf8').digest('hex');

    // Claim each request once, even if the same output is seen again.
    await fs.writeFile(pendingFile, Buffer.from(JSON.stringify({ id, requestHash, status: 'open' })));
    await fs.rename(pendingFile, stateFile, { overwrite: false });

    let response;
    try {
        const result = await showDialog(dialog, requestText, requestHash, requestFile, desktopPython);
        response = { id, requestHash, status: 'done', [dialog.resultKey]: result };
    } catch (error) {
        response = { id, requestHash, status: 'error', error: String(error.message || error) };
    }
    await fs.writeFile(pendingFile, Buffer.from(JSON.stringify(response)));
    await fs.rename(pendingFile, stateFile, { overwrite: true });
}

// =============================================================================
function activate(context) {
    let discoveredPython;
    const desktopPython = () => {
        const configured = context.globalState.get(desktopPythonKey);
        if (configured) return Promise.resolve(configured);
        discoveredPython ??= findDesktopPython().finally(() => { discoveredPython = undefined; });
        return discoveredPython;
    };
    context.subscriptions.push(vscode.commands.registerCommand('pglDialogs.selectDesktopPython', () => selectDesktopPython(context)));
    if (vscode.env.remoteName !== 'ssh-remote') return;

    const activeRequests = new Set();
    context.subscriptions.push(vscode.workspace.onDidChangeNotebookDocument(event => {
        const source = event.notebook.uri;
        if (source.scheme !== 'vscode-remote' || !source.authority.startsWith('ssh-remote+')) return;

        for (const change of event.cellChanges) {
            for (const output of change.outputs || []) {
                const item = output.items.find(item => item.mime === requestMime);
                if (!item || item.data.length > 128) continue;

                let id;
                try {
                    id = JSON.parse(Buffer.from(item.data).toString('utf8')).id;
                } catch {
                    continue;
                }
                if (typeof id !== 'string' || !/^[0-9a-f]{32}$/.test(id)) continue;
                const root = source.with({ path: `/tmp/pgl-dialog-${id}`, query: '', fragment: '' });
                const key = root.toString();
                if (activeRequests.has(key)) continue;
                activeRequests.add(key);
                serveRequest(root, id, desktopPython).catch(error => {
                    // Old outputs and interrupted cells can refer to removed requests.
                    if (!['FileNotFound', 'FileExists'].includes(error.code)) console.error('PGL Dialogs:', error);
                }).finally(() => activeRequests.delete(key));
            }
        }
    }));
}

// =============================================================================
// Run the bundled PGL dialog on the desktop
// =============================================================================

// =============================================================================
async function showDialog(dialog, requestText, requestHash, remoteRequest, desktopPython) {
    const python = await desktopPython();
    if (!python) {
        throw new Error('Desktop Python with PySide6 is required. Use PGL Dialogs: Select Desktop Python in the VS Code Command Palette.');
    }

    const directory = await localFs.mkdtemp(path.join(os.tmpdir(), 'pgl-dialog-'));
    const input = path.join(directory, 'request.json');
    const output = path.join(directory, 'response.json');
    const controller = new AbortController();
    let watcher;
    try {
        await localFs.writeFile(input, requestText, { encoding: 'utf8', mode: 0o600 });
        // An interrupted kernel removes the request. Stop its desktop process too.
        const checkRequest = async () => {
            try {
                await vscode.workspace.fs.stat(remoteRequest);
            } catch (error) {
                if (error.code === 'FileNotFound') controller.abort();
            }
            if (!controller.signal.aborted) watcher = setTimeout(checkRequest, 1000);
        };
        await checkRequest();
        await run(python, ['-I', path.join(__dirname, dialog.script), input, output], {
            signal: controller.signal, timeout: 900000, windowsHide: true,
        });
        const result = JSON.parse(await localFs.readFile(output, 'utf8'));
        if (dialog.resultKey !== 'confirmed') return result;
        if (result?.requestHash !== requestHash || typeof result.confirmed !== 'boolean') {
            throw new Error('Invalid desktop confirmation response.');
        }
        return result.confirmed;
    } catch (error) {
        if (controller.signal.aborted) throw vscode.FileSystemError.FileNotFound(remoteRequest);
        const details = String(error.stderr || error.message).trim().slice(-2000);
        throw new Error(`Could not run the PySide6 dialog on your desktop. ${details}\nUse PGL Dialogs: Select Desktop Python to choose an environment with PySide6.`);
    } finally {
        controller.abort();
        clearTimeout(watcher);
        await localFs.rm(directory, { recursive: true, force: true });
    }
}

// =============================================================================
// Find and select a local PySide6 environment
// =============================================================================

// =============================================================================
async function canUsePython(executable) {
    if (!path.isAbsolute(executable)) return false;
    try {
        await run(executable, ['-I', '-c', 'from PySide6 import QtWidgets'], { timeout: 4000, windowsHide: true });
        return true;
    } catch {
        return false;
    }
}

// =============================================================================
async function findDesktopPython() {
    const windows = process.platform === 'win32';
    const envFile = path.join(os.homedir(), '.conda', 'environments.txt');
    const registered = await localFs.readFile(envFile, 'utf8').catch(() => '');
    const environments = registered.split(/\r?\n/).map(prefix => prefix.trim()).filter(prefix => path.isAbsolute(prefix));
    const candidates = [
        ...environments.filter(prefix => /^pgl(?:$|_)/i.test(path.basename(prefix)))
            .map(prefix => path.join(prefix, ...(windows ? ['python.exe'] : ['bin', 'python']))),
        ...(process.env.PATH || '').split(path.delimiter).filter(prefix => path.isAbsolute(prefix))
            .flatMap(prefix => (windows ? ['python.exe'] : ['python3', 'python']).map(name => path.join(prefix, name))),
    ];
    for (const executable of new Set(candidates)) {
        if (await canUsePython(executable)) return executable;
    }
}

// =============================================================================
async function selectDesktopPython(context) {
    const executable = await vscode.window.showInputBox({
        title: 'PGL Dialogs',
        prompt: 'Enter the full path to a Python executable with PySide6 on this computer',
        value: context.globalState.get(desktopPythonKey) || '',
        validateInput: value => path.isAbsolute(value.trim()) ? undefined : 'Enter an absolute path on this computer.',
    });
    if (!executable) return;
    if (!await canUsePython(executable.trim())) {
        await vscode.window.showErrorMessage('Could not import PySide6 with that desktop Python. Choose a local Python environment with PySide6 installed.');
        return;
    }
    await context.globalState.update(desktopPythonKey, executable.trim());
}

// =============================================================================
// Validate the request before using the bundled dialog
// =============================================================================

// =============================================================================
function validConfirmation(request) {
    return request && ['title', 'summary', 'confirmButtonText'].every(key =>
        typeof request[key] === 'string' && request[key].trim() && !request[key].includes('\0')) &&
        ['details', 'copyText'].every(key =>
            request[key] === null || (typeof request[key] === 'string' && request[key].trim() && !request[key].includes('\0')));
}

// =============================================================================
function validSelection(request) {
    return request && typeof request.title === 'string' && request.title.trim() &&
        Array.isArray(request.items) && request.items.length > 0 &&
        request.items.every(label => typeof label === 'string');
}

module.exports = { activate };
