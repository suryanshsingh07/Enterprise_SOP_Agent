import React, { useState } from 'react';
import { Upload, FileUp, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadSOP } from '../services/api';

const UploadPanel = () => {
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState('idle'); // idle, uploading, success, error
    const [message, setMessage] = useState('');

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setStatus('idle');
            setMessage('');
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        
        setStatus('uploading');
        try {
            await uploadSOP(file);
            setStatus('success');
            setMessage('Document successfully uploaded and is being processed.');
            setFile(null);
        } catch (err) {
            setStatus('error');
            setMessage(err.message || 'Failed to upload document');
        }
    };

    return (
        <div className="glass p-8 rounded-2xl max-w-xl w-full mx-auto space-y-6">
            <div>
                <h3 className="text-xl font-semibold mb-1 text-white flex items-center gap-2">
                    <FileUp className="text-brand-400" /> Upload New SOP
                </h3>
                <p className="text-sm text-gray-400">Upload PDF documents to expand the knowledge base.</p>
            </div>

            <div className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-brand-500 transition-colors bg-gray-800/30">
                <input
                    type="file"
                    id="sop-upload"
                    accept=".pdf"
                    className="hidden"
                    onChange={handleFileChange}
                />
                <label htmlFor="sop-upload" className="cursor-pointer flex flex-col items-center gap-3">
                    <div className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center text-gray-400">
                        <Upload size={20} />
                    </div>
                    <div>
                        <span className="text-brand-400 font-medium hover:underline">Click to browse</span>
                        <span className="text-gray-400"> or drag and drop</span>
                    </div>
                    <p className="text-xs text-gray-500">PDF documents only (max 10MB)</p>
                </label>
            </div>

            {file && (
                <div className="flex items-center justify-between p-3 rounded-lg bg-gray-800/80 border border-gray-700">
                    <div className="truncate text-sm text-gray-300 font-medium">
                        {file.name}
                    </div>
                    <button
                        onClick={handleUpload}
                        disabled={status === 'uploading'}
                        className="px-4 py-1.5 bg-brand-600 hover:bg-brand-500 rounded-md text-sm font-medium transition-colors flex items-center gap-2 disabled:opacity-50"
                    >
                        {status === 'uploading' ? <Loader2 size={16} className="animate-spin" /> : 'Upload'}
                    </button>
                </div>
            )}

            {status === 'success' && (
                <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 flex items-center gap-2 text-sm">
                    <CheckCircle size={16} /> {message}
                </div>
            )}
            {status === 'error' && (
                <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 flex items-center gap-2 text-sm">
                    <AlertCircle size={16} /> {message}
                </div>
            )}
        </div>
    );
};

export default UploadPanel;
