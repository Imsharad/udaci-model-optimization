#!/usr/bin/env python3
"""
Cloud Run service for running model compression experiments.
Based on the headless compression script and adapted for serving.
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from project.starter_kit.run_compression_headless import run_compression_experiments

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Udacity Model Compression Service",
    description="Cloud service for running model compression experiments",
    version="1.0.0"
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
    "current_experiment": None,
    "status": "idle",
    "last_results": None,
    "start_time": None,
    "end_time": None
}

async def run_compression_experiments_async():
    """Run compression experiments asynchronously."""
    global experiment_status
    
    try:
        experiment_status["status"] = "running"
        experiment_status["start_time"] = datetime.utcnow().isoformat()
        
        logger.info("Starting compression experiments...")
        
        # Change to the correct directory before running the script
        original_cwd = os.getcwd()
        os.chdir('/app/project/starter_kit')

        # Run the synchronous function in a separate thread
        results = await asyncio.to_thread(run_compression_experiments)
        
        # Change back to the original directory
        os.chdir(original_cwd)

        experiment_status["status"] = "completed"
        experiment_status["end_time"] = datetime.utcnow().isoformat()
        experiment_status["last_results"] = results
        
        logger.info("All experiments completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Compression experiments failed: {e}")
        experiment_status["status"] = "failed"
        experiment_status["end_time"] = datetime.utcnow().isoformat()
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 Starting Model Compression Service")

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run and load balancers."""
    import torch
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_status": experiment_status["status"],
        "pytorch_available": True,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "version": "1.0.0"
    }
    
    return JSONResponse(content=health_data)

@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "Udacity Model Compression Service",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "experiments/start": "/experiments/start",
            "experiments/status": "/experiments/status",
            "experiments/results": "/experiments/results"
        }
    }

@app.post("/experiments/start")
async def start_experiments(background_tasks: BackgroundTasks):
    """Start compression experiments in the background."""
    if experiment_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Experiments already running")
    
    # Reset status
    experiment_status["status"] = "starting"
    experiment_status["last_results"] = None
    
    # Run experiments in background
    background_tasks.add_task(run_compression_experiments_async)
    
    return {
        "message": "Compression experiments started",
        "status": "starting",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/experiments/status")
async def get_experiment_status():
    """Get current experiment status."""
    return {
        "status": experiment_status["status"],
        "current_experiment": experiment_status["current_experiment"],
        "start_time": experiment_status["start_time"],
        "end_time": experiment_status["end_time"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/experiments/results")
async def get_experiment_results():
    """Get the results of the last completed experiments."""
    if experiment_status["last_results"] is None:
        raise HTTPException(status_code=404, detail="No experiment results available")
    
    return {
        "results": experiment_status["last_results"],
        "status": experiment_status["status"],
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    # Get port from environment (Cloud Run sets PORT)
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting compression service on port {port}")
    uvicorn.run(
        "compression-main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
