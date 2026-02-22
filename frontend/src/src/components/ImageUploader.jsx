import React, { useCallback, useState } from 'react'
import { UploadCloud, FileImage, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

export default function ImageUploader({ onUpload, isProcessing }) {
    const [dragActive, setDragActive] = useState(false)
    const [selectedFile, setSelectedFile] = useState(null)
    const [previewUrl, setPreviewUrl] = useState(null)

    const handleDrag = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true)
        } else if (e.type === "dragleave") {
            setDragActive(false)
        }
    }, [])

    const handleDrop = useCallback((e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0])
        }
    }, [])

    const handleChange = (e) => {
        e.preventDefault()
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0])
        }
    }

    const handleFile = (file) => {
        if (!file.type.startsWith('image/')) return
        setSelectedFile(file)
        const url = URL.createObjectURL(file)
        setPreviewUrl(url)
    }

    const clearFile = () => {
        setSelectedFile(null)
        setPreviewUrl(null)
    }

    const submit = () => {
        if (selectedFile && !isProcessing) {
            onUpload(selectedFile)
        }
    }

    return (
        <div className="h-full flex flex-col justify-center gap-4">
            <AnimatePresence mode="wait">
                {!previewUrl ? (
                    <motion.div
                        key="dropzone"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className={`relative flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-2xl h-full transition-all duration-300 ${dragActive
                                ? 'border-indigo-400 bg-indigo-500/10'
                                : 'border-slate-700 bg-slate-900/50 hover:bg-slate-800/80 hover:border-slate-500'
                            }`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                    >
                        <input
                            type="file"
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                            accept="image/*"
                            onChange={handleChange}
                        />

                        <div className={`p-4 rounded-full mb-4 transition-colors ${dragActive ? 'bg-indigo-500/20' : 'bg-slate-800'}`}>
                            <UploadCloud className={`w-8 h-8 ${dragActive ? 'text-indigo-400' : 'text-slate-400'}`} />
                        </div>

                        <p className="text-slate-200 font-medium mb-1">
                            Drag & Drop your math image
                        </p>
                        <p className="text-slate-500 text-sm">
                            or click to browse files
                        </p>
                    </motion.div>
                ) : (
                    <motion.div
                        key="preview"
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="flex flex-col h-full bg-slate-900/50 rounded-2xl border border-white/10 overflow-hidden"
                    >
                        <div className="flex-1 relative bg-black/40 flex items-center justify-center p-4">
                            <img
                                src={previewUrl}
                                alt="Preview"
                                className="max-h-[220px] max-w-full object-contain filter drop-shadow-[0_0_15px_rgba(255,255,255,0.1)] transition-transform hover:scale-105"
                            />
                            <button
                                onClick={clearFile}
                                className="absolute top-3 right-3 p-1.5 bg-black/50 hover:bg-red-500/80 text-white rounded-full backdrop-blur-md transition-colors z-20"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="p-4 bg-slate-900 border-t border-white/5 flex items-center justify-between">
                            <div className="flex items-center gap-2 overflow-hidden">
                                <FileImage className="w-4 h-4 text-indigo-400 shrink-0" />
                                <span className="text-sm text-slate-300 truncate">{selectedFile?.name}</span>
                            </div>
                            <button
                                onClick={submit}
                                disabled={isProcessing}
                                className="ml-4 px-6 py-2 bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-400 hover:to-blue-500 text-white text-sm font-medium rounded-xl shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isProcessing ? 'Processing...' : 'Analyze Physics'}
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}
