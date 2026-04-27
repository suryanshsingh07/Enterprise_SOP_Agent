const { searchRelevantChunks } = require('./vectorSearchService');
const { GoogleGenAI } = require('@google/genai');
const logger = require('../utils/logger');

const generateRAGResponse = async (queryText, apiKey, onChunk) => {
    try {
        const ai = new GoogleGenAI({ apiKey: apiKey || process.env.GEMINI_API_KEY });
        // 1. Retrieve Context
        const chunks = await searchRelevantChunks(queryText, apiKey);
        
        if (!chunks || chunks.length === 0) {
            onChunk("I don't know based on the provided SOPs.");
            return;
        }

        // 2. Build Context String
        let contextString = "";
        let sources = [];
        
        chunks.forEach((c, idx) => {
            contextString += `[Chunk ${idx + 1} - Source: ${c.documentName}, Page: ${c.page}, Section: ${c.section}]\n${c.text}\n\n`;
            sources.push({ documentName: c.documentName, page: c.page, section: c.section });
        });

        // 3. System Prompt Strict Rules
        const prompt = `You are an enterprise SOP assistant.

RULES:
- Answer ONLY from provided context
- If answer not found, say: "I don't know based on the provided SOPs."
- ALWAYS cite sources in the format: (Document Name, Page X, Section Y)

CONTEXT:
${contextString}

QUESTION:
${queryText}
`;

        // 4. Stream from Gemini
        const result = await ai.models.generateContentStream({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        for await (const chunk of result) {
            const chunkText = chunk.text;
            onChunk(chunkText);
        }

        // Send metadata as a final special chunk to the client
        onChunk(`\n\n__SOURCES__${JSON.stringify(sources)}`);
        
    } catch (error) {
        logger.error(`RAG Service Error: ${error.message}`);
        throw error;
    }
};

module.exports = { generateRAGResponse };
