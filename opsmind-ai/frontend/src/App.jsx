import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import { ChatProvider, ChatContext } from './context/ChatContext';
import { useContext } from 'react';
import ChatPage from './pages/ChatPage';
import AdminPage from './pages/AdminPage';
import LoginPage from './pages/LoginPage';

const ProtectedRoute = ({ children }) => {
  const { userEmail, apiKey } = useContext(ChatContext);
  const location = useLocation();
  if (!userEmail || !apiKey) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  return children;
};

const NavBar = () => {
  const { userEmail, logout } = useContext(ChatContext);
  return (
    <nav className="glass sticky top-0 z-50 flex justify-between items-center px-6 py-4">
      <div className="text-xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent flex items-center gap-4">
        OpsMind AI
      </div>
      <div className="space-x-6 flex items-center">
        <Link to="/" className="text-gray-300 hover:text-white transition-colors">Chat</Link>
        <Link to="/admin" className="text-gray-300 hover:text-white transition-colors">Uploads</Link>
        {userEmail && (
          <button onClick={logout} className="text-sm px-3 py-1 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg transition-colors">
            Logout ({userEmail.split('@')[0]})
          </button>
        )}
      </div>
    </nav>
  );
};

function App() {
  return (
    <Router>
      <ChatProvider>
        <div className="min-h-screen flex flex-col bg-gray-950">
          <NavBar />
          <main className="flex-1 overflow-hidden flex">
              <Routes>
                <Route path="/login" element={<LoginPage />} />
                <Route path="/" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
                <Route path="/admin" element={<ProtectedRoute><AdminPage /></ProtectedRoute>} />
              </Routes>
          </main>
        </div>
      </ChatProvider>
    </Router>
  );
}

export default App;
