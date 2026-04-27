const { generateRAGResponse } = require('../services/ragService');
const logger = require('../utils/logger');

const streamQuery = async (req, res) => {
    const { query } = req.body;
    const apiKey = req.headers['x-gemini-api-key'] || process.env.GEMINI_API_KEY;

    if (!query) {
        return res.status(400).json({ error: 'Query is required' });
    }

    // Setup SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders(); // flush the headers to establish SSE

    const onChunk = (textChunk) => {
        // SSE format
        res.write(`data: ${JSON.stringify({ text: textChunk })}\n\n`);
    };

    try {
        await generateRAGResponse(query, apiKey, onChunk);
        res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
        res.end();
    } catch (error) {
        logger.error(`Query Stream Error: ${error.message}`);
        res.write(`data: ${JSON.stringify({ error: 'Failed to process query' })}\n\n`);
        res.end();
    }
};

module.exports = { streamQuery };
