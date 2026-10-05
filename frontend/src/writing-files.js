// Plain-text interchange is separate from schema-1 resource packages. Imported
// files become new drafts; they never replace a loaded resource or its metadata.
export const writingFileLimit = 1024 * 1024;
const encoder = new TextEncoder();

function boundedTitle(value) {
  let title = '', bytes = 0;
  for (const character of value) {
    const length = encoder.encode(character).length;
    if (bytes + length > 1024) break;
    title += character; bytes += length;
  }
  return title || 'Untitled';
}

export function decodeWritingFile(bytes, name) {
  if (!/\.(md|markdown|txt)$/i.test(name)) throw new Error('Choose a Markdown (.md) or plain text (.txt) file.');
  if (bytes.byteLength > writingFileLimit) throw new Error('Writing files must be 1 MiB or smaller.');
  let content;
  try { content = new TextDecoder('utf-8', {fatal:true}).decode(bytes); }
  catch { throw new Error('This file is not valid UTF-8. Save it as UTF-8 text and try again.'); }
  if (content.includes('\0')) throw new Error('This file contains null characters. Choose a text file.');
  const title = boundedTitle(String(name).split(/[\\/]/).at(-1).replace(/\.(md|markdown|txt)$/i, '').trim());
  return {title, content};
}

export function writingDownloadName(title, extension) {
  if (!['md', 'txt'].includes(extension)) throw new Error('Unsupported text export format.');
  // File chooser names must also work on Windows. Preserve readable Unicode;
  // remove separators/control characters and reserved device basenames.
  let name = [...String(title || 'Untitled').replace(/[<>:"/\\|?*\u0000-\u001f\u007f]/g, '_')].slice(0, 100).join('').replace(/[. ]+$/g, '') || 'Untitled';
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(name)) name = 'Writing-' + name;
  return `${name}.${extension}`;
}

export function downloadWritingText(draft, extension) {
  const url = URL.createObjectURL(new Blob([draft.content], {type: extension === 'md' ? 'text/markdown;charset=utf-8' : 'text/plain;charset=utf-8'}));
  const link = document.createElement('a');
  link.href = url; link.download = writingDownloadName(draft.title, extension); link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
