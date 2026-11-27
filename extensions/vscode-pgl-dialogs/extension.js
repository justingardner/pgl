const vscode = require('vscode');
const { createHash, randomUUID } = require('node:crypto');

const requestMime = 'application/vnd.pgl.dialog+json';
const requestLimit = 64 * 1024;

// =============================================================================
// Receive notebook requests and send replies over VS Code's remote filesystem
// =============================================================================

// =============================================================================
async function serveRequest(root, id) {
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
    if (version !== 1 || mode !== 'confirm' || typeof requestText !== 'string') return;
    const request = JSON.parse(requestText);
    if (!validConfirmation(request)) return;
    const requestHash = createHash('sha256').update(requestText, 'utf8').digest('hex');

    // Claim each request once, even if the same output is seen again.
    await fs.writeFile(pendingFile, Buffer.from(JSON.stringify({ id, requestHash, status: 'open' })));
    await fs.rename(pendingFile, stateFile, { overwrite: false });

    let response;
    try {
        response = { id, requestHash, status: 'done', confirmed: await showConfirmation(request) };
    } catch (error) {
        response = { id, requestHash, status: 'error', error: String(error.message || error) };
    }
    await fs.writeFile(pendingFile, Buffer.from(JSON.stringify(response)));
    await fs.rename(pendingFile, stateFile, { overwrite: true });
}

// =============================================================================
function activate(context) {
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
                serveRequest(root, id).catch(error => {
                    // Old outputs and interrupted cells can refer to removed requests.
                    if (!['FileNotFound', 'FileExists'].includes(error.code)) console.error('PGL Dialogs:', error);
                }).finally(() => activeRequests.delete(key));
            }
        }
    }));
}

// =============================================================================
// Confirmation dialogs
// =============================================================================

// =============================================================================
function validConfirmation(request) {
    return request && ['title', 'summary', 'confirmButtonText'].every(key =>
        typeof request[key] === 'string' && request[key].trim() && !request[key].includes('\0')) &&
        ['details', 'copyText'].every(key =>
            request[key] === null || (typeof request[key] === 'string' && request[key].trim() && !request[key].includes('\0')));
}

// =============================================================================
async function showConfirmation(request) {
    const { title, summary, details, confirmButtonText, copyText } = request;
    const confirm = { title: confirmButtonText };
    const copy = { title: copyText };
    const buttons = copyText ? [confirm, copy] : [confirm];
    const options = { modal: true, detail: details ? `${summary}\n\n${details}` : summary };

    while (true) {
        const answer = await vscode.window.showInformationMessage(title, options, ...buttons);
        if (answer !== copy) return answer === confirm;
        await vscode.env.clipboard.writeText(details || summary);
    }
}

module.exports = { activate };
