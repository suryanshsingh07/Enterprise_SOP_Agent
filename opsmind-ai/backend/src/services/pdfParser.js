const fs = require('fs');
const pdf = require('pdf-parse');
const logger = require('../utils/logger');

const parsePDF = async (filePath) => {
    try {
        const dataBuffer = fs.readFileSync(filePath);
        const data = await pdf(dataBuffer);
        
        return {
            text: data.text,
            numpages: data.numpages
        };
    } catch (error) {
        logger.error(`Error parsing PDF: ${error.message}`);
        throw error;
    }
};

module.exports = { parsePDF };
