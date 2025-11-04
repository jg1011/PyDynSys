(function() {
    function renderMath() {
        if (typeof renderMathInElement !== 'undefined') {
            // Render notebook markdown cells with $ delimiters
            const notebookCells = document.querySelectorAll('.jp-RenderedMarkdown, .jp-MarkdownOutput');
            notebookCells.forEach(cell => {
                renderMathInElement(cell, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false}
                    ],
                    throwOnError: false,
                    strict: false
                });
            });
            
            // Render markdown files (which already have \(...\) from pymdownx.arithmatex)
            // Only render elements that are NOT inside notebook cells
            const markdownContainers = document.querySelectorAll('.md-content, article, main');
            markdownContainers.forEach(container => {
                // Skip if this container contains notebook cells
                if (!container.querySelector('.jp-RenderedMarkdown, .jp-MarkdownOutput')) {
                    renderMathInElement(container, {
                        delimiters: [
                            {left: '\\[', right: '\\]', display: true},
                            {left: '\\(', right: '\\)', display: false}
                        ],
                        throwOnError: false
                    });
                } else {
                    // Container has notebook cells - render only non-notebook content
                    const allTextElements = container.querySelectorAll('p, div, span, li, td, th, h1, h2, h3, h4, h5, h6');
                    allTextElements.forEach(element => {
                        if (!element.closest('.jp-RenderedMarkdown, .jp-MarkdownOutput')) {
                            renderMathInElement(element, {
                                delimiters: [
                                    {left: '\\[', right: '\\]', display: true},
                                    {left: '\\(', right: '\\)', display: false}
                                ],
                                throwOnError: false
                            });
                        }
                    });
                }
            });
        }
    }

    // Initial render on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(renderMath, 100);
        });
    } else {
        setTimeout(renderMath, 100);
    }

    // For Material's instant loading - subscribe to navigation events
    if (typeof document$ !== 'undefined') {
        document$.subscribe(function() {
            setTimeout(renderMath, 100);
        });
    }
})();