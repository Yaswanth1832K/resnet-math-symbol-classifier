import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { UploadCloud, PenTool, Sparkles, Wand2, Activity, Github } from 'lucide-react'
import ImageUploader from './components/ImageUploader'
import DrawingCanvas from './components/DrawingCanvas'
import ResultDisplay from './components/ResultDisplay'
import 'katex/dist/katex.min.css'

function App() {
    const [activeTab, setActiveTab] = useState('upload') // upload or draw
    const [isProcessing, setIsProcessing] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    const handlePredict = async (fileOrBlob) => {
        setIsProcessing(true)
        setError(null)
        setResult(null)

        const formData = new FormData()
        formData.append('file', fileOrBlob, 'image.png')

        try {
            // Pointing to our FastAPI backend
            const response = await fetch('https://resnet-math-symbol-classifier.onrender.com', {
                method: 'POST',
                body: formData,
            })

            const data = await response.json()

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Failed to process image')
            }

            setResult(data)
        } catch (err) {
            setError(err.message)
        } finally {
            setIsProcessing(false)
        }
    }

    return (
        <div className="min-h-screen bg-[#0a0a0a] text-slate-200 relative overflow-hidden font-sans selection:bg-indigo-500/30">

            {/* Dynamic Background Effects */}
            <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-900/20 blur-[120px] pointer-events-none" />
            <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-blue-900/20 blur-[120px] pointer-events-none" />

            {/* Grid Pattern overlay */}
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none mix-blend-overlay"></div>
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none"></div>

            {/* Header */}
            <header className="relative z-10 border-b border-white/5 bg-black/20 backdrop-blur-xl">
                <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-blue-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                                MathSymbol<span className="text-indigo-400">.AI</span>
                            </h1>
                            <p className="text-xs text-slate-500 font-medium tracking-wide uppercase">ResNet-18 Powered OCR</p>
                        </div>
                    </div>
                    <a href="https://github.com/Yaswanth1832K/resnet-math-symbol-classifier" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors bg-white/5 hover:bg-white/10 px-4 py-2 rounded-full border border-white/5">
                        <Github className="w-4 h-4" />
                        <span>View Source</span>
                    </a>
                </div>
            </header>

            {/* Main Content */}
            <main className="relative z-10 max-w-5xl mx-auto px-6 py-12 flex flex-col items-center">

                {/* Hero Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-center mb-12 max-w-2xl"
                >
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 text-indigo-300 text-xs font-medium mb-6 border border-indigo-500/20">
                        <Activity className="w-3 h-3 animate-pulse" />
                        Live Inference Engine Online
                    </div>
                    <h2 className="text-5xl font-bold mb-6 tracking-tight leading-tight">
                        Turn handwritten math into <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-blue-400 to-cyan-400 animate-gradient-x">
                            perfect LaTeX.
                        </span>
                    </h2>
                    <p className="text-slate-400 text-lg leading-relaxed">
                        Upload an image or draw directly on the screen. Our fine-tuned ResNet-18 model instantly segments and classifies your symbols with high precision.
                    </p>
                </motion.div>

                <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

                    {/* Left Column: Input Panel */}
                    <motion.div
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 }}
                        className="flex flex-col bg-white/[0.02] border border-white/5 rounded-3xl overflow-hidden backdrop-blur-sm shadow-2xl relative"
                    >
                        {/* Input Tabs */}
                        <div className="flex border-b border-white/5 bg-black/20 p-2 gap-2">
                            <button
                                onClick={() => setActiveTab('upload')}
                                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl text-sm font-medium transition-all ${activeTab === 'upload'
                                        ? 'bg-white/10 text-white shadow-sm'
                                        : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
                                    }`}
                            >
                                <UploadCloud className="w-4 h-4" />
                                Upload Image
                            </button>
                            <button
                                onClick={() => setActiveTab('draw')}
                                className={`flex-1 flex items-center justify-center gap-2 py-3 rounded-2xl text-sm font-medium transition-all ${activeTab === 'draw'
                                        ? 'bg-white/10 text-white shadow-sm'
                                        : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
                                    }`}
                            >
                                <PenTool className="w-4 h-4" />
                                Draw Symbol
                            </button>
                        </div>

                        {/* Input Area */}
                        <div className="p-6 h-[400px]">
                            <AnimatePresence mode="wait">
                                {activeTab === 'upload' ? (
                                    <motion.div
                                        key="upload"
                                        initial={{ opacity: 0, scale: 0.98 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.98 }}
                                        transition={{ duration: 0.2 }}
                                        className="h-full"
                                    >
                                        <ImageUploader onUpload={handlePredict} isProcessing={isProcessing} />
                                    </motion.div>
                                ) : (
                                    <motion.div
                                        key="draw"
                                        initial={{ opacity: 0, scale: 0.98 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.98 }}
                                        transition={{ duration: 0.2 }}
                                        className="h-full flex flex-col"
                                    >
                                        <DrawingCanvas onPredict={handlePredict} isProcessing={isProcessing} />
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </div>

                        {/* Overlay Loading State */}
                        {isProcessing && (
                            <div className="absolute inset-0 z-20 bg-black/60 backdrop-blur-md flex flex-col items-center justify-center rounded-3xl">
                                <div className="w-16 h-16 relative flex items-center justify-center">
                                    <div className="absolute inset-0 rounded-full border-t-2 border-indigo-500 animate-spin"></div>
                                    <div className="absolute inset-2 rounded-full border-r-2 border-blue-400 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                                    <Wand2 className="w-6 h-6 text-indigo-400 animate-pulse" />
                                </div>
                                <p className="mt-4 text-indigo-200 font-medium">Running Deep Learning Model...</p>
                                <p className="text-xs text-slate-400 mt-2">Segmenting image & predicting symbols</p>
                            </div>
                        )}
                    </motion.div>

                    {/* Right Column: Results Panel */}
                    <motion.div
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                        className="flex flex-col h-full min-h-[500px]"
                    >
                        <ResultDisplay result={result} error={error} isProcessing={isProcessing} />
                    </motion.div>

                </div>
            </main>

            {/* Footer */}
            <footer className="relative z-10 border-t border-white/5 py-8 mt-12 text-center text-sm text-slate-500">
                <p>Powered by ResNet-18 • FastAPI • PyTorch • React</p>
            </footer>
        </div>
    )
}

export default App
