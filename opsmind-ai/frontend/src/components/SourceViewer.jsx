import React from 'react';
import { BookOpen, FileText } from 'lucide-react';

const SourceViewer = ({ sources }) => {
    if (!sources || sources.length === 0) return null;

    return (
        <div className="mt-3 pt-3 border-t border-gray-700/50">
            <h4 className="text-xs font-semibold text-gray-400 mb-2 flex items-center gap-1.5">
                <BookOpen size={12} /> Sources Cited
            </h4>
            <div className="flex flex-wrap gap-2">
                {sources.map((source, idx) => (
                    <div key={idx} className="flex items-center gap-2 bg-gray-800/80 px-2.5 py-1.5 rounded text-xs text-blue-300 border border-blue-900/30 hover:border-blue-500/50 cursor-pointer transition-colors">
                        <FileText size={12} />
                        <span>{source.documentName}</span>
                        <span className="text-gray-500">•</span>
                        <span>Pg {source.page}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default SourceViewer;
