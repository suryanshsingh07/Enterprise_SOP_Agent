import React from 'react';
import { User, Bot } from 'lucide-react';
import SourceViewer from './SourceViewer';

const MessageBubble = ({ message }) => {
    const isUser = message.sender === 'user';

    return (
        <div className={`flex gap-4 p-4 rounded-xl animate-fade-in ${isUser ? 'bg-brand-600/10 ml-auto max-w-[85%]' : 'glass max-w-[95%]'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${isUser ? 'bg-brand-500 text-white' : 'bg-indigo-500 text-white'}`}>
                {isUser ? <User size={18} /> : <Bot size={18} />}
            </div>
            <div className="flex-1 space-y-2 overflow-hidden">
                <div className="text-sm font-medium text-gray-300">
                    {isUser ? 'You' : 'OpsMind Agent'}
                </div>
                <div className="text-gray-100 whitespace-pre-wrap leading-relaxed text-sm">
                    {message.text}
                </div>
                {!isUser && <SourceViewer sources={message.sources} />}
            </div>
        </div>
    );
};

export default MessageBubble;
