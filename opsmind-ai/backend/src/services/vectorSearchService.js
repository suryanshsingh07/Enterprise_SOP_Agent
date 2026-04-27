const Chunk = require('../models/Chunk');
const { generateEmbedding } = require('./embeddingService');
const { VECTOR_INDEX_NAME, SEARCH_LIMIT } = require('../config/vector');
const logger = require('../utils/logger');

const searchRelevantChunks = async (queryText, apiKey) => {
    try {
        const queryEmbedding = await generateEmbedding(queryText, apiKey);

        const results = await Chunk.aggregate([
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": queryEmbedding,
                    "numCandidates": 100,
                    "limit": SEARCH_LIMIT
                }
            },
            {
                "$lookup": {
                    "from": "documents",
                    "localField": "documentId",
                    "foreignField": "_id",
                    "as": "document"
                }
            },
            {
                "$unwind": "$document"
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "page": 1,
                    "section": 1,
                    "score": { "$meta": "vectorSearchScore" },
                    "documentName": "$document.originalName"
                }
            }
        ]);

        return results;
    } catch (error) {
        logger.error(`Vector search error: ${error.message}`);
        throw error;
    }
};

module.exports = { searchRelevantChunks };
