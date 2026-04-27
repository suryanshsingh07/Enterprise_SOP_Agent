const express = require('express');
const multer = require('multer');
const path = require('path');
const { uploadDocument, getDocuments } = require('../controllers/adminController');

const router = express.Router();

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + path.extname(file.originalname))
    }
});

const upload = multer({ storage: storage });

router.post('/upload', upload.single('document'), uploadDocument);
router.get('/documents', getDocuments);

module.exports = router;
