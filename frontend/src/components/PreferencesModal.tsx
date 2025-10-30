import React, { useState, useEffect, useCallback } from 'react';
import { User, Sparkles, Eye, SlidersHorizontal } from 'lucide-react';
import websocketService, { MessageType } from '../services/websocket';

interface PreferencesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ModelOption {
  id: string;
  label: string;
  description?: string;
}

const PreferencesModal: React.FC<PreferencesModalProps> = ({ isOpen, onClose }) => {
  const [systemPrompt, setSystemPrompt] = useState('');
  const [userName, setUserName] = useState('');
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'profile' | 'system' | 'models'>('profile');
  const [isVisionEnabled, setIsVisionEnabled] = useState(false);
  const [sttOptions, setSttOptions] = useState<ModelOption[]>([]);
  const [ttsOptions, setTtsOptions] = useState<ModelOption[]>([]);
  const [currentSttModel, setCurrentSttModel] = useState('');
  const [currentTtsModel, setCurrentTtsModel] = useState('');
  const [selectedSttModel, setSelectedSttModel] = useState('');
  const [selectedTtsModel, setSelectedTtsModel] = useState('');
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';

  const fetchModelOptions = useCallback(async () => {
    setModelsLoading(true);
    setModelsError(null);

    try {
      const response = await fetch(`${apiBaseUrl}/models`);

      if (!response.ok) {
        throw new Error(`Failed to load models: ${response.status}`);
      }

      const data = await response.json();
      const stt = data?.stt ?? {};
      const tts = data?.tts ?? {};

      const sttList: ModelOption[] = Array.isArray(stt.options) ? stt.options : [];
      const ttsList: ModelOption[] = Array.isArray(tts.options) ? tts.options : [];

      setSttOptions(sttList);
      setTtsOptions(ttsList);

      const sttCurrent = typeof stt.current === 'string' ? stt.current : '';
      const ttsCurrent = typeof tts.current === 'string' ? tts.current : '';

      setCurrentSttModel(sttCurrent);
      setCurrentTtsModel(ttsCurrent);
      setSelectedSttModel(sttCurrent);
      setSelectedTtsModel(ttsCurrent);
    } catch (error) {
      console.error('Error loading model options', error);
      setModelsError('Failed to load available models. Ensure the backend is running.');
    } finally {
      setModelsLoading(false);
    }
  }, [apiBaseUrl]);
  
  useEffect(() => {
    if (isOpen) {
      // Reset state when modal opens
      setSaveError(null);
      
      // Fetch current system prompt, user profile, and vision settings
      const handleSystemPrompt = (data: any) => {
        if (data && data.prompt) {
          setSystemPrompt(data.prompt);
        }
      };
      
      const handleUserProfile = (data: any) => {
        if (data && data.name !== undefined) {
          setUserName(data.name);
        }
      };
      
      const handleVisionSettings = (data: any) => {
        if (data && data.enabled !== undefined) {
          setIsVisionEnabled(data.enabled);
        }
      };
      
      // Listen for responses
      websocketService.addEventListener(MessageType.SYSTEM_PROMPT, handleSystemPrompt);
      websocketService.addEventListener(MessageType.USER_PROFILE, handleUserProfile);
      websocketService.addEventListener(MessageType.VISION_SETTINGS, handleVisionSettings);
      
      // Request data
      websocketService.getSystemPrompt();
      websocketService.getUserProfile();
      websocketService.getVisionSettings();

      fetchModelOptions();

      console.log('Requested preferences data');
      
      return () => {
        websocketService.removeEventListener(MessageType.SYSTEM_PROMPT, handleSystemPrompt);
        websocketService.removeEventListener(MessageType.USER_PROFILE, handleUserProfile);
        websocketService.removeEventListener(MessageType.VISION_SETTINGS, handleVisionSettings);
      };
    }
  }, [isOpen, fetchModelOptions]);
  
  // Listen for update confirmations
  useEffect(() => {
    if (activeTab === 'models') {
      return;
    }

    let updateCount = 0;
    const expectedUpdateCount = 3; // Always expect 3 updates: system prompt, user profile, and vision
    let success = true;
    
    const handlePromptUpdated = (data: any) => {
      updateCount++;
      if (!(data && data.success)) {
        success = false;
        setSaveError('Failed to update system prompt. Please try again.');
      }
      
      if (updateCount >= expectedUpdateCount) {
        setIsSaving(false);
        if (success) {
          // Close modal only if all updates succeeded
          onClose();
        }
      }
    };
    
    const handleProfileUpdated = (data: any) => {
      updateCount++;
      if (!(data && data.success)) {
        success = false;
        setSaveError('Failed to update user profile. Please try again.');
      }
      
      if (updateCount >= expectedUpdateCount) {
        setIsSaving(false);
        if (success) {
          // Close modal only if all updates succeeded
          onClose();
        }
      }
    };
    
    const handleVisionSettingsUpdated = (data: any) => {
      updateCount++;
      if (!(data && data.success)) {
        success = false;
        setSaveError('Failed to update vision settings. Please try again.');
      }
      
      if (updateCount >= expectedUpdateCount) {
        setIsSaving(false);
        if (success) {
          // Close modal only if all updates succeeded
          onClose();
        }
      }
    };
    
    websocketService.addEventListener(MessageType.SYSTEM_PROMPT_UPDATED, handlePromptUpdated);
    websocketService.addEventListener(MessageType.USER_PROFILE_UPDATED, handleProfileUpdated);
    websocketService.addEventListener(MessageType.VISION_SETTINGS_UPDATED, handleVisionSettingsUpdated);
    
    return () => {
      websocketService.removeEventListener(MessageType.SYSTEM_PROMPT_UPDATED, handlePromptUpdated);
      websocketService.removeEventListener(MessageType.USER_PROFILE_UPDATED, handleProfileUpdated);
      websocketService.removeEventListener(MessageType.VISION_SETTINGS_UPDATED, handleVisionSettingsUpdated);
    };
  }, [onClose, activeTab]);
  
  const handleSave = async () => {
    if (activeTab === 'models') {
      setSaveError(null);
      setModelsError(null);

      const payload: Record<string, unknown> = {};

      if (selectedSttModel && selectedSttModel !== currentSttModel) {
        payload.stt_model_id = selectedSttModel;
      }

      if (selectedTtsModel && selectedTtsModel !== currentTtsModel) {
        payload.tts_model_id = selectedTtsModel;
      }

      if (Object.keys(payload).length === 0) {
        onClose();
        return;
      }

      setIsSaving(true);

      try {
        const response = await fetch(`${apiBaseUrl}/models/select`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          const message = await response.text();
          throw new Error(message || 'Failed to update models');
        }

        await response.json();

        if (payload.stt_model_id) {
          const sttId = payload.stt_model_id as string;
          setCurrentSttModel(sttId);
          setSelectedSttModel(sttId);
        }

        if (payload.tts_model_id) {
          const ttsId = payload.tts_model_id as string;
          setCurrentTtsModel(ttsId);
          setSelectedTtsModel(ttsId);
        }

        await fetchModelOptions();
        onClose();
      } catch (error) {
        console.error('Failed to update models', error);
        setModelsError('Failed to update models. Check backend logs for more details.');
      } finally {
        setIsSaving(false);
      }

      return;
    }

    // Check if system prompt is empty when in system tab
    if (activeTab === 'system' && !systemPrompt.trim()) {
      setSaveError('System prompt cannot be empty');
      return;
    }

    setIsSaving(true);
    setSaveError(null);

    // Always update all settings
    websocketService.updateSystemPrompt(systemPrompt);
    websocketService.updateUserProfile(userName);
    websocketService.updateVisionSettings(isVisionEnabled);
  };
  
  // backticks, in my code, in the year of our lord, 2025? no.
  const handleRestore = () => { 
    setSystemPrompt(
      "You are a helpful, friendly, and concise voice assistant. " +
      "Respond to user queries in a natural, conversational manner. " +
      "Keep responses brief and to the point, as you're communicating via voice. " +
      "When providing information, focus on the most relevant details. " +
      "If you don't know something, admit it rather than making up an answer." +
      "\n\n" +
      "Through the webapp, you can receive and understand photographs and pictures." +
      "\n\n" +
      "When the user sends a message like '[silent]', '[no response]', or '[still waiting]', it means they've gone quiet or haven't responded." +
      "When you see these signals, continue the conversation naturally based on the previous topic and context." +
      "Stay on topic, be helpful, and don't mention that they were silent - just carry on the conversation as if you're gently following up."
    );
  };
  
  const renderProfileTab = () => (
    <div className="space-y-4">
      <div className="space-y-2">
        <label className="text-sm font-medium text-slate-300">Your Name</label>
        <div className="relative">
          <input
            value={userName}
            onChange={(e) => setUserName(e.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700 rounded-lg pl-10 p-3 text-slate-300 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            placeholder="Enter your name (optional)"
          />
          <User className="absolute left-3 top-3 w-4 h-4 text-slate-500" />
        </div>
        <p className="text-xs text-slate-400">
          Your name will be used to personalize greetings and make the conversation feel more natural.
        </p>
      </div>
    </div>
  );
  
  const renderSystemTab = () => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-slate-300">System Prompt</label>
        <button
          onClick={handleRestore}
          className="text-xs text-sky-400 hover:text-sky-300"
        >
          Restore Default
        </button>
      </div>
      <textarea
        value={systemPrompt}
        onChange={(e) => setSystemPrompt(e.target.value)}
        className="w-full h-64 bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-slate-300 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-emerald-500"
        placeholder="Enter system prompt..."
      />
      <p className="text-xs text-slate-400">
        The system prompt defines how the AI assistant behaves when responding to your voice commands.
      </p>
    </div>
  );

  const renderModelsTab = () => {
    const sttDetail = sttOptions.find((option) => option.id === selectedSttModel);
    const ttsDetail = ttsOptions.find((option) => option.id === selectedTtsModel);

    return (
      <div className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300">Speech-to-Text Model</label>
          <select
            value={selectedSttModel}
            onChange={(event) => setSelectedSttModel(event.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-slate-300 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            disabled={modelsLoading || sttOptions.length === 0}
          >
            {sttOptions.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-slate-400 min-h-[2.5rem]">
            {sttDetail?.description ?? 'Select the speech recognition model used for live transcription.'}
          </p>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-300">Text-to-Speech Model</label>
          <select
            value={selectedTtsModel}
            onChange={(event) => setSelectedTtsModel(event.target.value)}
            className="w-full bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-slate-300 text-sm focus:outline-none focus:ring-1 focus:ring-emerald-500"
            disabled={modelsLoading || ttsOptions.length === 0}
          >
            {ttsOptions.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-slate-400 min-h-[2.5rem]">
            {ttsDetail?.description ?? 'Choose the voice synthesis engine used for playback.'}
          </p>
        </div>

        <div className="p-3 bg-slate-800/30 border border-slate-700/60 rounded-lg text-xs text-slate-300">
          <p>
            Model changes reload the backend pipelines and may take a few moments. Sessions in progress should be paused before
            switching for best results.
          </p>
        </div>
      </div>
    );
  };
  
  // Handle animation state
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    if (isOpen) {
      setIsVisible(true);
    } else {
      setTimeout(() => setIsVisible(false), 300); // Match animation duration
    }
  }, [isOpen]);
  
  if (!isOpen && !isVisible) return null;
  
  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm transition-opacity duration-300 ease-in-out ${isOpen ? 'opacity-100 bg-black/50' : 'opacity-0 pointer-events-none'}`}>
      <div className={`bg-slate-900 border border-slate-700 rounded-lg w-full max-w-2xl max-h-[90vh] flex flex-col shadow-xl transition-all duration-300 ease-in-out ${isOpen ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
        {/* Header */}
        <div className="flex items-center p-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-slate-100">Preferences</h2>
        </div>
        
        {/* Tabs */}
        <div className="flex border-b border-slate-700">
          <button
            className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
              activeTab === 'profile'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }`}
            onClick={() => setActiveTab('profile')}
          >
            <User className="w-4 h-4" />
            <span>User Profile</span>
          </button>
          <button
            className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
              activeTab === 'system'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }`}
            onClick={() => setActiveTab('system')}
          >
            <Sparkles className="w-4 h-4" />
            <span>System Prompt</span>
          </button>
          <button
            className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
              activeTab === 'models'
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-slate-400 hover:text-slate-300'
            }`}
            onClick={() => setActiveTab('models')}
          >
            <SlidersHorizontal className="w-4 h-4" />
            <span>Models</span>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          {activeTab === 'profile' && renderProfileTab()}
          {activeTab === 'system' && renderSystemTab()}
          {activeTab === 'models' && renderModelsTab()}

          {activeTab !== 'models' && saveError && (
            <div className="text-red-400 text-sm p-2 bg-red-900/20 border border-red-900/30 rounded">
              {saveError}
            </div>
          )}

          {activeTab === 'models' && modelsError && (
            <div className="text-red-400 text-sm p-2 bg-red-900/20 border border-red-900/30 rounded">
              {modelsError}
            </div>
          )}

          {activeTab === 'models' && modelsLoading && (
            <div className="text-xs text-slate-400">Loading available models…</div>
          )}
        </div>
        
        {/* Vision Settings Section */}
        <div className="px-4 py-3 border-t border-slate-700 bg-slate-800/30">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Eye className="w-4 h-4 text-indigo-400" />
              <span className="text-sm font-medium text-slate-300">Vision</span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isVisionEnabled ? 'bg-indigo-600' : 'bg-slate-700'
                }`}
                onClick={() => setIsVisionEnabled(!isVisionEnabled)}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    isVisionEnabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </div>
              <span className="text-xs text-slate-400">
                {isVisionEnabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            When enabled, Vocalis can analyze images and provide visual context (coming soon).
          </p>
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-slate-700 flex justify-end space-x-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className={`
              px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg
              ${isSaving ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            {isSaving ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PreferencesModal;
