import React, { useEffect, useState } from 'react';
import ChatWindow from '../components/ChatWindow';
import { getDocuments } from '../services/api';
import { FileText, Clock } from 'lucide-react';

const ChatPage = () => {
    const [documents, setDocuments] = useState([]);
    const [loading, setLoading] = useState(true);

    const fetchDocs = async () => {
        try {
            const data = await getDocuments();
            setDocuments(data);
        } catch (error) {
            console.error('Failed to fetch documents history', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchDocs();
    }, []);

    // We pass fetchDocs to ChatWindow so it can refresh history after a new upload
    return (
        <div className="w-full h-full flex bg-cover bg-center overflow-hidden" style={{ backgroundImage: 'radial-gradient(circle at top right, rgba(37, 99, 235, 0.1), transparent 40%)' }}>
            {/* Sidebar */}
            <div className="w-64 glass border-r border-gray-800/50 flex flex-col hidden md:flex">
                <div className="p-4 border-b border-gray-800/50">
                    <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                        <Clock size={16} /> Document History
                    </h2>
                </div>
                <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin">
                    {loading ? (
                        <div className="text-xs text-gray-500 animate-pulse">Loading history...</div>
                    ) : documents.length === 0 ? (
                        <div className="text-xs text-gray-500">No documents uploaded yet.</div>
                    ) : (
                        documents.map((doc) => (
                            <div key={doc._id} className="p-3 rounded-xl bg-gray-900/40 border border-gray-800 hover:border-gray-700 transition-colors flex items-start gap-3">
                                <div className="mt-0.5 text-brand-400">
                                    <FileText size={16} />
                                </div>
                                <div className="overflow-hidden">
                                    <div className="text-sm font-medium text-gray-200 truncate" title={doc.originalName}>
                                        {doc.originalName}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1 flex items-center gap-2">
                                        <span className={doc.status === 'completed' ? 'text-green-500/80' : doc.status === 'failed' ? 'text-red-500/80' : 'text-yellow-500/80'}>
                                            ● {doc.status}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col items-center justify-center relative">
                <ChatWindow onUploadSuccess={fetchDocs} />
            </div>
        </div>
    );
};

export default ChatPage;
