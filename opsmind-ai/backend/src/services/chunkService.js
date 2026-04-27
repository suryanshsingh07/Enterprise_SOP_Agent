const chunkText = (text, chunkSize = 1000, chunkOverlap = 100) => {
    const chunks = [];
    let i = 0;
    while (i < text.length) {
        let chunk = text.slice(i, i + chunkSize);
        chunks.push(chunk);
        i += chunkSize - chunkOverlap;
    }
    return chunks;
};

module.exports = { chunkText };
