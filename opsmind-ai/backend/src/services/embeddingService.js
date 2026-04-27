const { GoogleGenAI } = require('@google/genai');
const logger = require('../utils/logger');

const generateEmbedding = async (text, apiKey) => {
    try {
        const ai = new GoogleGenAI({ apiKey: apiKey || process.env.GEMINI_API_KEY });
        const response = await ai.models.embedContent({
            model: 'gemini-embedding-2',
            contents: text,
        });
        return response.embeddings[0].values;
    } catch (error) {
        logger.error(`Error generating embedding: ${error.message}`);
        throw error;
    }
};

module.exports = { generateEmbedding };
