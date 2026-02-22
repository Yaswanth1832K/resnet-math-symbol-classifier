import React, { useRef, useState, useEffect } from 'react'
import { Eraser, Undo2, Play } from 'lucide-react'

export default function DrawingCanvas({ onPredict, isProcessing }) {
    const canvasRef = useRef(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [history, setHistory] = useState([]) // For undo functionality

    // Set up high-DPI canvas
    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')

        // Fill with white background initially since model expects white bg/black ink (or vice versa handled by adaptive thresholding)
        // Actually our dataset typically has black pen on white. Let's stick to white canvas, black pen.
        ctx.fillStyle = '#ffffff'
        ctx.fillRect(0, 0, canvas.width, canvas.height)

    }, [])

    const startDrawing = (e) => {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const rect = canvas.getBoundingClientRect()

        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height

        const clientX = e.clientX ? e.clientX : e.touches[0].clientX
        const clientY = e.clientY ? e.clientY : e.touches[0].clientY

        const x = (clientX - rect.left) * scaleX
        const y = (clientY - rect.top) * scaleY

        // Save state for undo before starting a new stroke
        setHistory([...history, canvas.toDataURL()])

        ctx.beginPath()
        ctx.moveTo(x, y)
        setIsDrawing(true)
    }

    const draw = (e) => {
        if (!isDrawing) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const rect = canvas.getBoundingClientRect()

        const scaleX = canvas.width / rect.width
        const scaleY = canvas.height / rect.height

        const clientX = e.clientX ? e.clientX : e.touches[0].clientX
        const clientY = e.clientY ? e.clientY : e.touches[0].clientY

        const x = (clientX - rect.left) * scaleX
        const y = (clientY - rect.top) * scaleY

        ctx.lineTo(x, y)
        ctx.strokeStyle = '#000000'
        ctx.lineWidth = 4
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
        ctx.stroke()
    }

    const stopDrawing = () => {
        setIsDrawing(false)
    }

    const clearCanvas = () => {
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        setHistory([...history, canvas.toDataURL()])
        ctx.fillStyle = '#ffffff'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
    }

    const undo = () => {
        if (history.length === 0) return

        const previousState = history[history.length - 1]
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')

        const img = new Image()
        img.src = previousState
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height)
            ctx.drawImage(img, 0, 0)
        }

        setHistory(history.slice(0, -1))
    }

    const submitDrawing = () => {
        if (isProcessing) return
        const canvas = canvasRef.current

        // Convert canvas to blob
        canvas.toBlob((blob) => {
            onPredict(blob)
        }, 'image/png')
    }

    return (
        <div className="h-full flex flex-col pt-4">
            {/* Tools */}
            <div className="flex items-center justify-between mb-3 px-1">
                <div className="flex gap-2">
                    <button
                        onClick={undo}
                        disabled={history.length === 0}
                        className="p-2 rounded-lg bg-slate-800/50 text-slate-400 hover:bg-slate-700 hover:text-white disabled:opacity-30 transition-colors"
                        title="Undo"
                    >
                        <Undo2 className="w-4 h-4" />
                    </button>
                    <button
                        onClick={clearCanvas}
                        className="p-2 rounded-lg bg-slate-800/50 text-slate-400 hover:bg-red-500/20 hover:text-red-400 transition-colors"
                        title="Clear"
                    >
                        <Eraser className="w-4 h-4" />
                    </button>
                </div>
                <p className="text-xs text-slate-500 font-medium">Draw horizontally and clearly</p>
            </div>

            {/* Canvas Wrapper */}
            <div className="flex-1 bg-white rounded-xl overflow-hidden border-2 border-slate-700 shadow-inner mb-4 relative cursor-crosshair">
                <canvas
                    ref={canvasRef}
                    width={800} // Logical width (scaled via css)
                    height={300}
                    className="w-full h-full object-contain"
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseOut={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                />
            </div>

            <button
                onClick={submitDrawing}
                disabled={isProcessing}
                className="w-full py-3 flex items-center justify-center gap-2 bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-400 hover:to-blue-500 text-white rounded-xl shadow-[0_0_20px_rgba(99,102,241,0.2)] font-medium transition-all disabled:opacity-50"
            >
                <Play className="w-4 h-4" />
                {isProcessing ? 'Analyzing...' : 'Analyze Drawing'}
            </button>

        </div>
    )
}
