import React, { useContext, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChatContext } from '../context/ChatContext';
import { Key, Mail, LogIn, ChevronRight } from 'lucide-react';

const LoginPage = () => {
    const { userEmail, login, apiKey, saveApiKey } = useContext(ChatContext);
    const [step, setStep] = useState(userEmail ? 2 : 1);
    const [emailInput, setEmailInput] = useState(userEmail || '');
    const [keyInput, setKeyInput] = useState(apiKey || '');
    const navigate = useNavigate();

    const handleLogin = (e) => {
        e.preventDefault();
        if (emailInput.trim()) {
            login(emailInput.trim());
            setStep(2);
        }
    };

    const handleKeySubmit = (e) => {
        e.preventDefault();
        if (keyInput.trim()) {
            saveApiKey(keyInput.trim());
            navigate('/');
        }
    };

    return (
        <div className="w-full h-full flex items-center justify-center bg-gray-950 p-6">
            <div className="max-w-md w-full glass p-8 rounded-2xl relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
                
                <div className="text-center mb-8">
                    <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent mb-2">
                        Welcome to OpsMind
                    </h2>
                    <p className="text-gray-400 text-sm">
                        {step === 1 ? 'Sign in to access your enterprise knowledge' : 'Connect your Gemini API Key to continue'}
                    </p>
                </div>

                {step === 1 ? (
                    <form onSubmit={handleLogin} className="space-y-6 flex flex-col items-center">
                        <div className="w-full relative">
                            <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Email Address</label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                                <input
                                    type="email"
                                    value={emailInput}
                                    onChange={(e) => setEmailInput(e.target.value)}
                                    placeholder="Enter your email ID"
                                    className="w-full bg-gray-900/50 border border-gray-700/50 text-gray-200 rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent transition-all"
                                    required
                                />
                            </div>
                        </div>
                        <button
                            type="submit"
                            className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white rounded-xl font-medium shadow-lg shadow-blue-500/20 transition-all flex items-center justify-center gap-2 group"
                        >
                            Continue <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
                        </button>
                    </form>
                ) : (
                    <form onSubmit={handleKeySubmit} className="space-y-6 flex flex-col items-center">
                        <div className="w-full relative">
                            <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Gemini API Key</label>
                            <div className="relative">
                                <Key className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
                                <input
                                    type="password"
                                    value={keyInput}
                                    onChange={(e) => setKeyInput(e.target.value)}
                                    placeholder="Enter API key"
                                    className="w-full bg-gray-900/50 border border-gray-700/50 text-gray-200 rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-transparent transition-all"
                                    required
                                />
                            </div>
                            <p className="text-xs text-gray-500 mt-2">Your key is stored locally in your browser.</p>
                        </div>
                        <div className="flex gap-3 w-full">
                            <button
                                type="button"
                                onClick={() => setStep(1)}
                                className="w-1/3 py-3 px-4 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-xl font-medium transition-all"
                            >
                                Back
                            </button>
                            <button
                                type="submit"
                                className="w-2/3 py-3 px-4 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white rounded-xl font-medium shadow-lg shadow-indigo-500/20 transition-all flex items-center justify-center gap-2"
                            >
                                Start Chatting <LogIn size={18} />
                            </button>
                        </div>
                    </form>
                )}
            </div>
        </div>
    );
};

export default LoginPage;
