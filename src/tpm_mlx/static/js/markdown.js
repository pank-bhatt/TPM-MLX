/**
 * TPM-MLX Markdown & Reasoning Thought Formatter
 * Sanitized Markdown renderer supporting tables, headers, lists, code blocks,
 * and collapsible internal reasoning thoughts (<think> blocks).
 */

/**
 * Escapes unsafe HTML characters to prevent XSS.
 * @param {string} text
 * @returns {string}
 */
export function escapeHtml(text) {
    if (!text) return "";
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Toggles a collapsible reasoning thought block.
 * @param {HTMLElement} header
 */
export function toggleThought(header) {
    if (!header || !header.parentElement) return;
    header.parentElement.classList.toggle("collapsed");
}

/**
 * Parses markdown into safe semantic HTML.
 * @param {string} text
 * @returns {string}
 */
export function parseMarkdown(text) {
    if (!text) return "";
    const codeBlocks = [];
    let processedText = text;
    
    // 1. Extract and format multi-line code blocks
    processedText = processedText.replace(/```([\s\S]*?)```/g, (match, codeBlock) => {
        const lineBreakIndex = codeBlock.indexOf("\n");
        let lang = "code";
        let code = codeBlock;
        if (lineBreakIndex !== -1) {
            lang = codeBlock.substring(0, lineBreakIndex).trim();
            code = codeBlock.substring(lineBreakIndex + 1);
        }
        const escapedCode = escapeHtml(code);
        const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
        codeBlocks.push(`<pre><code class="language-${lang}">${escapedCode}</code></pre>`);
        return placeholder;
    });

    // 2. Extract and format inline code
    const inlineCodes = [];
    processedText = processedText.replace(/`([^`\n]+)`/g, (match, code) => {
        const placeholder = `__INLINE_CODE_${inlineCodes.length}__`;
        inlineCodes.push(`<code>${escapeHtml(code)}</code>`);
        return placeholder;
    });

    // 3. Escape all remaining raw HTML tags
    processedText = escapeHtml(processedText);

    // Helper to check for delimiter rows in tables
    function isDelimiterRow(row) {
        if (!row) return false;
        const t = row.trim();
        return t.startsWith('|') && /^[|:\-\s]+$/.test(t);
    }

    // 4. Parse block elements line by line
    const lines = processedText.split('\n');
    let inList = false;
    let listType = null;
    let inTable = false;
    let tableRows = [];
    let tableAlignments = [];
    const result = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        // Headers
        const headerMatch = line.match(/^(#{1,6})\s+(.+)$/);
        if (headerMatch) {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
                listType = null;
            }
            if (inTable) {
                tableRows.push('</tbody>');
                result.push('<table>' + tableRows.join('\n') + '</table>');
                inTable = false;
                tableRows = [];
                tableAlignments = [];
            }
            const level = headerMatch[1].length;
            result.push(`<h${level}>${headerMatch[2]}</h${level}>`);
            continue;
        }

        // Horizontal Rules
        if (trimmed === '---' || trimmed === '***' || trimmed === '___') {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
                listType = null;
            }
            if (inTable) {
                tableRows.push('</tbody>');
                result.push('<table>' + tableRows.join('\n') + '</table>');
                inTable = false;
                tableRows = [];
                tableAlignments = [];
            }
            result.push('<hr>');
            continue;
        }

        // Unordered lists
        const ulMatch = line.match(/^(\s*)([-*+])\s+(.+)$/);
        if (ulMatch) {
            if (inTable) {
                tableRows.push('</tbody>');
                result.push('<table>' + tableRows.join('\n') + '</table>');
                inTable = false;
                tableRows = [];
                tableAlignments = [];
            }
            if (!inList || listType !== 'ul') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ul>');
                inList = true;
                listType = 'ul';
            }
            result.push(`<li>${ulMatch[3]}</li>`);
            continue;
        }

        // Ordered lists
        const olMatch = line.match(/^(\s*)(\d+)\.\s+(.+)$/);
        if (olMatch) {
            if (inTable) {
                tableRows.push('</tbody>');
                result.push('<table>' + tableRows.join('\n') + '</table>');
                inTable = false;
                tableRows = [];
                tableAlignments = [];
            }
            if (!inList || listType !== 'ol') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ol>');
                inList = true;
                listType = 'ol';
            }
            result.push(`<li>${olMatch[3]}</li>`);
            continue;
        }

        // Table parsing check
        if (!inTable && line.includes('|') && i + 1 < lines.length && isDelimiterRow(lines[i + 1])) {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
                listType = null;
            }
            inTable = true;
            tableRows = [];
            tableAlignments = [];

            // Parse alignments from the delimiter row lines[i+1]
            const delimiterCols = lines[i + 1].split('|').map(s => s.trim());
            const startDelim = lines[i + 1].startsWith('|') ? 1 : 0;
            const endDelim = lines[i + 1].endsWith('|') ? delimiterCols.length - 1 : delimiterCols.length;
            const cleanDelim = delimiterCols.slice(startDelim, endDelim);

            tableAlignments = cleanDelim.map(col => {
                const start = col.startsWith(':');
                const end = col.endsWith(':');
                if (start && end) return 'center';
                if (end) return 'right';
                return 'left';
            });

            // Parse header row
            const headerCols = line.split('|').map(s => s.trim());
            const startIdx = line.startsWith('|') ? 1 : 0;
            const endIdx = line.endsWith('|') ? headerCols.length - 1 : headerCols.length;
            const cleanHeaders = headerCols.slice(startIdx, endIdx);

            tableRows.push('<thead><tr>' + cleanHeaders.map((col, idx) => {
                const align = tableAlignments[idx] || 'left';
                return `<th style="text-align: ${align}">${col}</th>`;
            }).join('') + '</tr></thead><tbody>');

            i++; // skip the delimiter row
            continue;
        }

        if (inTable) {
            if (line.includes('|')) {
                const cols = line.split('|').map(s => s.trim());
                const startIdx = line.startsWith('|') ? 1 : 0;
                const endIdx = line.endsWith('|') ? cols.length - 1 : cols.length;
                const cleanCols = cols.slice(startIdx, endIdx);

                tableRows.push('<tr>' + cleanCols.map((col, idx) => {
                    const align = tableAlignments[idx] || 'left';
                    return `<td style="text-align: ${align}">${col}</td>`;
                }).join('') + '</tr>');
                continue;
            } else {
                // Table ended
                tableRows.push('</tbody>');
                result.push('<table>' + tableRows.join('\n') + '</table>');
                inTable = false;
                tableRows = [];
                tableAlignments = [];
            }
        }

        // Empty lines close lists
        if (trimmed === '') {
            if (inList) {
                result.push(`</${listType}>`);
                inList = false;
                listType = null;
            }
            result.push('<br>');
            continue;
        }

        // Standard paragraph text
        if (inList) {
            result[result.length - 1] = result[result.length - 1].replace(/<\/li>$/, ` ${trimmed}</li>`);
        } else {
            result.push(`<p>${line}</p>`);
        }
    }

    if (inList) {
        result.push(`</${listType}>`);
    }
    if (inTable) {
        tableRows.push('</tbody>');
        result.push('<table>' + tableRows.join('\n') + '</table>');
    }

    let parsedHTML = result.join('\n');

    // 5. Parse inline tags (bold, italic, links)
    parsedHTML = parsedHTML.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    parsedHTML = parsedHTML.replace(/__(.*?)__/g, '<strong>$1</strong>');
    parsedHTML = parsedHTML.replace(/\*(.*?)\*/g, '<em>$1</em>');
    parsedHTML = parsedHTML.replace(/_(.*?)_/g, '<em>$1</em>');

    // Links (sanitized against javascript: and data: XSS vectors)
    parsedHTML = parsedHTML.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, linkText, url) => {
        const trimmedUrl = url.trim().toLowerCase();
        if (trimmedUrl.startsWith('javascript:') || trimmedUrl.startsWith('data:')) {
            return `<span>${linkText}</span>`;
        }
        const escapedUrl = url.replace(/"/g, '&quot;');
        return `<a href="${escapedUrl}" target="_blank" rel="noopener noreferrer">${linkText}</a>`;
    });

    // Restore inline code segments
    for (let i = 0; i < inlineCodes.length; i++) {
        parsedHTML = parsedHTML.replace(`__INLINE_CODE_${i}__`, inlineCodes[i]);
    }

    // Restore code blocks
    for (let i = 0; i < codeBlocks.length; i++) {
        parsedHTML = parsedHTML.replace(`__CODE_BLOCK_${i}__`, codeBlocks[i]);
    }

    // Cleanup duplicate breaks inside paragraphs
    parsedHTML = parsedHTML.replace(/<p><br><\/p>/g, '<br>');
    
    return parsedHTML;
}

/**
 * Formats a full assistant message, parsing reasoning <think> blocks and markdown.
 * @param {string} text
 * @returns {string}
 */
export function formatMessageHtml(text) {
    let htmlContent = "";
    let remainingText = text;

    // Check for reasoning think blocks (<think> or gemma-4 channel format)
    let thinkStart = text.indexOf("<think>");
    let thinkEnd = text.indexOf("</think>");
    let tagLenStart = 7;
    let tagLenEnd = 8;

    if (thinkStart === -1) {
        thinkStart = text.indexOf("<|channel>thought");
        thinkEnd = text.indexOf("<channel|>");
        tagLenStart = 17;
        tagLenEnd = 10;
    }

    if (thinkStart !== -1) {
        const before = text.substring(0, thinkStart);
        if (before) {
            htmlContent += `<div class="content-block">${parseMarkdown(before)}</div>`;
        }

        let thinking = "";
        if (thinkEnd !== -1) {
            thinking = text.substring(thinkStart + tagLenStart, thinkEnd);
            remainingText = text.substring(thinkEnd + tagLenEnd);
        } else {
            thinking = text.substring(thinkStart + tagLenStart);
            remainingText = "";
        }

        htmlContent += `
            <div class="thought-container">
                <div class="thought-header" onclick="toggleThought(this)">
                    <span>🧠 THOUGHT PROCESS</span>
                    <span class="thought-arrow">▼</span>
                </div>
                <div class="thought-content">${escapeHtml(thinking)}</div>
            </div>
        `;
    }

    if (remainingText) {
        htmlContent += `<div class="content-block">${parseMarkdown(remainingText)}</div>`;
    }

    return htmlContent;
}
