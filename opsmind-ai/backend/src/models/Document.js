const mongoose = require('mongoose');

const documentSchema = new mongoose.Schema({
    filename: { type: String, required: true },
    originalName: { type: String, required: true },
    uploadDate: { type: Date, default: Date.now },
    status: { type: String, enum: ['uploaded', 'processing', 'completed', 'failed'], default: 'uploaded' },
    errorMessage: { type: String }
});

module.exports = mongoose.model('Document', documentSchema);
