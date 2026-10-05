// Native titles are bounded in UTF-8 bytes, while HTML maxlength counts UTF-16
// units. Derived/imported titles must stop at complete Unicode code points.
const encoder = new TextEncoder();
export function boundedResourceTitle(value) {
  let title='', bytes=0;
  for (const character of String(value)) {
    const length=encoder.encode(character).length;
    if (bytes+length>1024) break;
    title+=character; bytes+=length;
  }
  return title || 'Untitled';
}
