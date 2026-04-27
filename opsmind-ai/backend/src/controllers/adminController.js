const Document = require('../models/Document');
const Chunk = require('../models/Chunk');
const { parsePDF } = require('../services/pdfParser');
const { chunkText } = require('../services/chunkService');
const { generateEmbedding } = require('../services/embeddingService');
const officeParser = require('officeparser');
const logger = require('../utils/logger');

const uploadDocument = async (req, res) => {
    if (!req.file) {
        res.status(400);
        throw new Error('No file uploaded');
    }

    try {
        const doc = await Document.create({
            filename: req.file.filename,
            originalName: req.file.originalname,
            status: 'processing'
        });

        const apiKey = req.headers['x-gemini-api-key'] || process.env.GEMINI_API_KEY;

        // Async processing
        processDocument(doc._id, req.file.path, apiKey).catch(err => {
            logger.error(`Failed processing doc ${doc._id}: ${err.message}`);
        });

        res.status(201).json({ message: 'Document uploaded and processing started', documentId: doc._id });
    } catch (error) {
        res.status(500);
        throw new Error('Server error during upload');
    }
};

const processDocument = async (docId, filePath, apiKey) => {
    try {
        let text = '';
        let totalPages = 1;
        
        if (filePath.toLowerCase().endsWith('.pdf')) {
            const parsed = await parsePDF(filePath);
            text = parsed.text;
            totalPages = parsed.numpages || 1;
        } else {
            text = await officeParser.parseOfficeAsync(filePath);
        }

        const chunks = chunkText(text, 1000, 100);
        
        for (let i = 0; i < chunks.length; i++) {
            const chunkStr = chunks[i];
            const embedding = await generateEmbedding(chunkStr, apiKey);
            
            await Chunk.create({
                text: chunkStr,
                embedding: embedding,
                documentId: docId,
                page: Math.max(1, Math.floor((i / chunks.length) * totalPages)), 
                section: `Section ${i+1}`
            });
        }

        await Document.findByIdAndUpdate(docId, { status: 'completed' });
        logger.info(`Document ${docId} processed successfully. ${chunks.length} chunks generated.`);
    } catch (error) {
        await Document.findByIdAndUpdate(docId, { status: 'failed', errorMessage: error.message });
        throw error;
    }
};

const getDocuments = async (req, res) => {
    try {
        const docs = await Document.find().sort({ createdAt: -1 });
        res.json(docs);
    } catch (error) {
        res.status(500).json({ error: 'Server error fetching documents' });
    }
};

module.exports = { uploadDocument, getDocuments };
