import { createContext, useState } from 'react';

export const ChatContext = createContext();

export const ChatProvider = ({ children }) => {
    const [messages, setMessages] = useState([]);
    const [isTyping, setIsTyping] = useState(false);
    
    // Auth state
    const [userEmail, setUserEmail] = useState(() => localStorage.getItem('opsmind_user_email') || null);
    const [apiKey, setApiKey] = useState(() => localStorage.getItem('opsmind_gemini_key') || null);

    const login = (email) => {
        setUserEmail(email);
        localStorage.setItem('opsmind_user_email', email);
    };

    const saveApiKey = (key) => {
        setApiKey(key);
        localStorage.setItem('opsmind_gemini_key', key);
    };

    const logout = () => {
        setUserEmail(null);
        setApiKey(null);
        localStorage.removeItem('opsmind_user_email');
        localStorage.removeItem('opsmind_gemini_key');
    };

    const addMessage = (message) => {
        setMessages(prev => [...prev, message]);
    };

    const updateLastMessage = (text, sources) => {
        setMessages(prev => {
            const newMessages = [...prev];
            const lastIdx = newMessages.length - 1;
            if (lastIdx >= 0 && newMessages[lastIdx].sender === 'ai') {
                newMessages[lastIdx] = { 
                    ...newMessages[lastIdx], 
                    text: text, 
                    sources: sources || newMessages[lastIdx].sources 
                };
            }
            return newMessages;
        });
    };

    return (
        <ChatContext.Provider value={{ 
            messages, addMessage, updateLastMessage, isTyping, setIsTyping,
            userEmail, apiKey, login, saveApiKey, logout
        }}>
            {children}
        </ChatContext.Provider>
    );
};
