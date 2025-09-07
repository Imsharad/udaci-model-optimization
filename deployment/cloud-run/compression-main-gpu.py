#!/usr/bin/env python3
"""
GPU-enabled Cloud Run service for aggressive model compression.
Enhanced version targeting 70% compression with GPU acceleration.
"""

import os
import json
import sys
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to python path
sys.path.append('/app/project/starter_kit')

from aggressive_compression_pipeline import run_aggressive_compression_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Udacity GPU Model Compression Service",
    description="GPU-accelerated model compression for 70% size reduction target",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global experiment status
experiment_status = {
    "current_stage": None,
    "status": "idle",
    "last_results": None,
    "start_time": None,
    "end_time": None,
    "progress": 0,
    "error_message": None
}

async def run_aggressive_compression_async():
    """Run aggressive compression pipeline asynchronously."""
    global experiment_status
    
    try:
        experiment_status["status"] = "running"
        experiment_status["start_time"] = datetime.utcnow().isoformat()
        experiment_status["current_stage"] = "initialization"
        experiment_status["progress"] = 0
        experiment_status["error_message"] = None
        
        logger.info("Starting aggressive compression pipeline...")
        
        # Update progress during execution
        experiment_status["current_stage"] = "baseline_loading"
        experiment_status["progress"] = 10
        
        # Run the compression pipeline
        results = await asyncio.to_thread(run_aggressive_compression_pipeline)
        
        experiment_status["status"] = "completed"
        experiment_status["end_time"] = datetime.utcnow().isoformat()
        experiment_status["last_results"] = results
        experiment_status["progress"] = 100
        experiment_status["current_stage"] = "completed"
        
        logger.info("Aggressive compression pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Compression pipeline failed: {e}")
        experiment_status["status"] = "failed"
        experiment_status["end_time"] = datetime.utcnow().isoformat()
        experiment_status["error_message"] = str(e)
        experiment_status["current_stage"] = "failed"
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 Starting GPU Model Compression Service v2.0")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU Available: {gpu_name} (Count: {gpu_count})")
        else:
            logger.warning("⚠️ No GPU detected, falling back to CPU")
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")

@app.get("/health")
async def health_check():
    """Enhanced health check with GPU information."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "N/A"
        memory_allocated = torch.cuda.memory_allocated(0) if gpu_available else 0
        memory_cached = torch.cuda.memory_reserved(0) if gpu_available else 0
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        gpu_available = False
        gpu_count = 0
        gpu_name = "Error"
        memory_allocated = 0
        memory_cached = 0
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_status": experiment_status["status"],
        "current_stage": experiment_status["current_stage"],
        "progress": experiment_status["progress"],
        "gpu_info": {
            "available": gpu_available,
            "count": gpu_count,
            "name": gpu_name,
            "memory_allocated_mb": memory_allocated / 1024 / 1024,
            "memory_cached_mb": memory_cached / 1024 / 1024
        },
        "version": "2.0.0",
        "service_type": "aggressive_compression"
    }
    
    return JSONResponse(content=health_data)

@app.get("/")
async def root():
    """Root endpoint with enhanced service information."""
    return {
        "message": "Udacity GPU Model Compression Service v2.0",
        "status": "running",
        "features": [
            "70% size reduction target",
            "GPU acceleration",
            "Multi-stage pipeline",
            "Real-time progress tracking"
        ],
        "endpoints": {
            "health": "/health",
            "compression/start": "/compression/start",
            "compression/status": "/compression/status",
            "compression/results": "/compression/results",
            "compression/progress": "/compression/progress"
        }
    }

@app.post("/compression/start")
async def start_compression(background_tasks: BackgroundTasks):
    """Start aggressive compression pipeline."""
    if experiment_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Compression already running")
    
    # Reset status
    experiment_status["status"] = "starting"
    experiment_status["last_results"] = None
    experiment_status["progress"] = 0
    experiment_status["current_stage"] = "queued"
    experiment_status["error_message"] = None
    
    # Run compression in background
    background_tasks.add_task(run_aggressive_compression_async)
    
    return {
        "message": "Aggressive compression pipeline started",
        "target": "70% size reduction, 60% speed improvement",
        "status": "starting",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/compression/status")
async def get_compression_status():
    """Get detailed compression status."""
    return {
        "status": experiment_status["status"],
        "current_stage": experiment_status["current_stage"],
        "progress": experiment_status["progress"],
        "start_time": experiment_status["start_time"],
        "end_time": experiment_status["end_time"],
        "error_message": experiment_status["error_message"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/compression/progress")
async def get_compression_progress():
    """Get real-time progress information."""
    stages = [
        "initialization",
        "baseline_loading", 
        "pruning",
        "quantization",
        "graph_optimization",
        "evaluation",
        "completed"
    ]
    
    current_stage_idx = 0
    if experiment_status["current_stage"] in stages:
        current_stage_idx = stages.index(experiment_status["current_stage"])
    
    return {
        "overall_progress": experiment_status["progress"],
        "current_stage": experiment_status["current_stage"],
        "stage_progress": (current_stage_idx + 1) / len(stages) * 100,
        "stages": stages,
        "status": experiment_status["status"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/compression/results")
async def get_compression_results():
    """Get comprehensive compression results."""
    if experiment_status["last_results"] is None:
        raise HTTPException(status_code=404, detail="No compression results available")
    
    results = experiment_status["last_results"]
    
    # Add summary for easy consumption
    summary = {
        "success": experiment_status["status"] == "completed",
        "targets_achieved": results.get("target_achievement", {}),
        "compression_summary": {
            "size_reduction": f"{results['compression_results']['size_reduction_percent']:.1f}%",
            "speed_improvement": f"{results['compression_results']['speed_improvement_percent']:.1f}%",
            "accuracy_drop": f"{results['compression_results']['accuracy_drop_percent']:.1f}%",
            "final_size_mb": results['compression_results']['final_size_mb'],
            "final_accuracy": f"{results['compression_results']['final_accuracy']:.2f}%"
        }
    }
    
    return {
        "summary": summary,
        "detailed_results": results,
        "status": experiment_status["status"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/compression/download-model")
async def download_compressed_model():
    """Provide download link for the compressed model."""
    if experiment_status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Compression not completed")
    
    model_path = "/app/project/starter_kit/models/aggressive_compression_pipeline/final_optimized_model.pth"
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Compressed model file not found")
    
    return {
        "model_path": model_path,
        "model_size_mb": os.path.getsize(model_path) / 1024 / 1024,
        "download_ready": True,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting GPU compression service on port {port}")
    uvicorn.run(
        "compression-main-gpu:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )