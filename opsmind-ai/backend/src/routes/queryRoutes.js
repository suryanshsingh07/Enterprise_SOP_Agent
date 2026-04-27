const express = require('express');
const { streamQuery } = require('../controllers/queryController');

const router = express.Router();

router.post('/stream', streamQuery);

module.exports = router;
