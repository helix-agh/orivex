window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    // Material's document$ subscription also handles the initial page load.
    typeset: false,
    pageReady: () => window.MathJax.startup.defaultPageReady().then(() => {
      document$.subscribe(() => {
        window.MathJax.startup.promise = window.MathJax.startup.promise.then(() => {
          window.MathJax.typesetClear();
          window.MathJax.texReset();
          return window.MathJax.typesetPromise();
        });
      });
    })
  }
};
