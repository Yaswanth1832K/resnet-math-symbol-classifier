import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertCircle, ChevronRight, Calculator, RefreshCw } from 'lucide-react'
import { BlockMath } from 'react-katex'

export default function ResultDisplay({ result, error, isProcessing }) {

    if (!result && !error && !isProcessing) {
        return (
            <div className="flex-1 flex flex-col items-center justify-center p-8 bg-white/[0.02] border border-white/5 rounded-3xl backdrop-blur-sm text-center">
                <div className="w-16 h-16 rounded-2xl bg-slate-900 flex items-center justify-center mb-6 border border-white/5 shadow-inner">
                    <Calculator className="w-8 h-8 text-slate-600" />
                </div>
                <h3 className="text-xl font-medium text-slate-300 mb-2">Awaiting Image</h3>
                <p className="text-slate-500 max-w-sm">
                    Upload an image or draw a math symbol. The results and LaTeX equation will appear here.
                </p>
            </div>
        )
    }

    if (error) {
        return (
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex-1 flex flex-col items-center justify-center p-8 bg-red-950/20 border border-red-500/20 rounded-3xl text-center"
            >
                <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-6">
                    <AlertCircle className="w-8 h-8 text-red-400" />
                </div>
                <h3 className="text-xl font-medium text-red-200 mb-2">Processing Error</h3>
                <p className="text-red-400/70">{error}</p>
                <button
                    onClick={() => window.location.reload()}
                    className="mt-6 flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-300 rounded-lg transition-colors border border-red-500/20"
                >
                    <RefreshCw className="w-4 h-4" /> Try Again
                </button>
            </motion.div>
        )
    }

    return (
        <div className="flex-1 flex flex-col gap-6">
            <AnimatePresence>
                {result && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex-1 flex flex-col gap-6"
                    >
                        {/* Final Equation Rendering */}
                        <div className="bg-gradient-to-br from-indigo-950/40 to-slate-900/60 border border-indigo-500/20 rounded-3xl p-8 relative overflow-hidden group">
                            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-400 via-indigo-500 to-purple-500"></div>

                            <h3 className="text-sm font-medium text-indigo-300 mb-6 flex items-center gap-2 uppercase tracking-widest">
                                <Calculator className="w-4 h-4" /> Final Equation
                            </h3>

                            <div className="flex items-center justify-center min-h-[140px] bg-black/40 rounded-2xl border border-white/5 shadow-inner p-4 overflow-x-auto">
                                <div className="text-2xl md:text-4xl text-white">
                                    <BlockMath math={result.latex || "\\text{No symbols analyzed}"} />
                                </div>
                            </div>

                            <div className="mt-6 flex items-center gap-4 bg-slate-900/50 rounded-xl p-4 border border-white/5">
                                <div className="px-3 py-1 bg-white/5 rounded-md text-slate-400 text-xs font-mono border border-white/10">
                                    LaTeX Code
                                </div>
                                <code className="text-sm text-cyan-300 font-mono flex-1 overflow-x-auto whitespace-nowrap scrollbar-hide">
                                    {result.latex}
                                </code>
                            </div>
                        </div>

                        {/* Segmentation Breakdown */}
                        <div className="bg-white/[0.02] border border-white/5 rounded-3xl p-6 flex-1">
                            <h3 className="text-sm font-medium text-slate-400 mb-6 flex items-center gap-2">
                                Confidence Breakdown
                            </h3>

                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                                {result.predictions?.map((pred, i) => (
                                    <motion.div
                                        key={i}
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        transition={{ delay: i * 0.1 }}
                                        className="flex flex-col items-center bg-slate-900/50 rounded-2xl p-4 border border-white/5 hover:border-indigo-500/30 transition-colors group relative overflow-hidden"
                                    >
                                        <div className="absolute inset-0 bg-gradient-to-b from-indigo-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                                        <div className="w-16 h-16 bg-white object-contain rounded-xl p-1 mb-4 shadow-inner">
                                            <img src={`data:image/png;base64,${pred.image_base64}`} alt={`Symbol ${i}`} className="w-full h-full object-contain filter invert" />
                                        </div>

                                        <div className="flex items-center justify-between w-full mt-auto">
                                            <span className="text-xs text-slate-500 uppercase font-medium">Class</span>
                                            <span className="text-lg font-bold text-white bg-white/10 px-2 py-0.5 rounded-md">{pred.class}</span>
                                        </div>
                                    </motion.div>
                                ))}
                            </div>
                        </div>

                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
