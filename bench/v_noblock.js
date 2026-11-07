//js: drop every rule whose selector names a code block body (all of #16865's cap styling at once)
() => {
  let n = 0;
  const walk = (list, owner) => {
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const r = list[i];
      if (r.cssRules && !r.selectorText) walk(r.cssRules, r);
      else if (r.selectorText && r.selectorText.includes('code-block-body')) { owner.deleteRule(i); n += 1; }
    }
  };
  for (const sheet of document.styleSheets) { try { walk(sheet.cssRules, sheet); } catch (e) {} }
  return n;
}
