//js: drop the custom ::-webkit-scrollbar rules (#16865 hover-only code scrollbars)
() => {
  let n = 0;
  const walk = (list, owner) => {
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const r = list[i];
      if (r.cssRules && !r.selectorText) walk(r.cssRules, r);
      else if (r.selectorText && r.selectorText.includes("::-webkit-scrollbar")) { owner.deleteRule(i); n += 1; }
    }
  };
  for (const sheet of document.styleSheets) { try { walk(sheet.cssRules, sheet); } catch (e) {} }
  return n;
}
