import React, { useEffect, useRef, useState, useCallback } from 'react';
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { Activity, RefreshCw, Camera, CameraOff, Info, Settings } from 'lucide-react';

export default function KowtowCounter() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [count, setCount] = useState(0);
  const [isBowed, setIsBowed] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const requestRef = useRef<number>();
  const stateRef = useRef<'STANDING' | 'PROSTRATING'>('STANDING');
  const minNoseYRef = useRef(1.0);
  const maxNoseYRef = useRef(0.0);
  const lastVideoTimeRef = useRef(-1);

  // Initialize MediaPipe Pose
  useEffect(() => {
    let active = true;
    const initModel = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          '/wasm'
        );
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/models/pose_landmarker_lite.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numPoses: 1,
        });
        if (active) {
          poseLandmarkerRef.current = poseLandmarker;
          setIsReady(true);
        }
      } catch (err) {
        if (active) {
          console.error(err);
          setError('加载AI模型失败，请检查网络连接。');
        }
      }
    };
    initModel();
    
    // Initial device enumeration (labels might be empty if no permission yet)
    navigator.mediaDevices.enumerateDevices().then(allDevices => {
      const videoDevices = allDevices.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      if (videoDevices.length > 0 && active) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    });

    return () => {
      active = false;
      if (poseLandmarkerRef.current) {
        poseLandmarkerRef.current.close();
      }
    };
  }, []);

  const renderLoop = useCallback(() => {
    if (!isRunning || !videoRef.current || !canvasRef.current || !poseLandmarkerRef.current) {
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (video.currentTime !== lastVideoTimeRef.current && video.readyState >= 2) {
      lastVideoTimeRef.current = video.currentTime;
      
      // Make sure canvas dimensions match video
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const results = poseLandmarkerRef.current.detectForVideo(video, performance.now());

      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (results.landmarks && results.landmarks.length > 0) {
          const drawingUtils = new DrawingUtils(ctx);
          for (const landmark of results.landmarks) {
            drawingUtils.drawLandmarks(landmark, {
              radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
              color: '#10b981',
              lineWidth: 2
            });
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
              color: '#ffffff',
              lineWidth: 2
            });
          }

          // Kowtow (大拜) logic
          const landmarks = results.landmarks[0];
          const nose = landmarks[0];
          const leftShoulder = landmarks[11];
          const rightShoulder = landmarks[12];

          if (nose && leftShoulder && rightShoulder) {
            const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
            
            // Slow decay to adapt to camera movements
            minNoseYRef.current = Math.min(1.0, minNoseYRef.current + 0.0001);
            maxNoseYRef.current = Math.max(0.0, maxNoseYRef.current - 0.0001);

            // Update min/max
            if (nose.y < minNoseYRef.current) minNoseYRef.current = nose.y;
            if (nose.y > maxNoseYRef.current) maxNoseYRef.current = nose.y;

            const amplitude = maxNoseYRef.current - minNoseYRef.current;

            // Only process if there's a reasonable range of motion (at least 15% of frame)
            if (amplitude > 0.15) {
              const standingThreshold = minNoseYRef.current + amplitude * 0.25;
              const prostratingThreshold = maxNoseYRef.current - amplitude * 0.25;

              const isHeadDown = nose.y > shoulderY - 0.05;
              const isHeadUp = nose.y < shoulderY + 0.05;

              if (stateRef.current === 'STANDING') {
                if (nose.y > prostratingThreshold && isHeadDown) {
                  stateRef.current = 'PROSTRATING';
                  setIsBowed(true);
                }
              } else if (stateRef.current === 'PROSTRATING') {
                if (nose.y < standingThreshold && isHeadUp) {
                  stateRef.current = 'STANDING';
                  setIsBowed(false);
                  setCount(c => c + 1);
                }
              }
            }
          }
        }
      }
    }
    
    requestRef.current = requestAnimationFrame(renderLoop);
  }, [isRunning]);

  useEffect(() => {
    if (isRunning) {
      requestRef.current = requestAnimationFrame(renderLoop);
    } else if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
    };
  }, [isRunning, renderLoop]);

  const startCameraWithId = async (deviceId: string) => {
    setError(null);
    if (!videoRef.current) return;
    
    // Stop existing stream if any
    if (videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play();
        setIsRunning(true);
      };

      // Re-enumerate devices to get proper labels now that we have permission
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = allDevices.filter(device => device.kind === 'videoinput');
      setDevices(videoDevices);
      
    } catch (err) {
      console.error(err);
      setError('无法访问摄像头，请授予权限或选择其他摄像头。');
      setIsRunning(false);
    }
  };

  const startCamera = () => startCameraWithId(selectedDeviceId);

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsRunning(false);
  };

  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newDeviceId = e.target.value;
    setSelectedDeviceId(newDeviceId);
    if (isRunning) {
      startCameraWithId(newDeviceId);
    }
  };

  const resetCount = () => {
    setCount(0);
    stateRef.current = 'STANDING';
    setIsBowed(false);
    minNoseYRef.current = 1.0;
    maxNoseYRef.current = 0.0;
  };

  return (
    <div className="min-h-screen bg-stone-900 text-stone-100 font-sans flex flex-col items-center py-8 px-4">
      <div className="max-w-3xl w-full space-y-8">
        
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight text-emerald-400">
            大拜计数器
          </h1>
          <p className="text-stone-400 text-lg">
            基于AI的礼佛大拜/磕头追踪器
          </p>
        </div>

        {/* Main Display */}
        <div className="bg-stone-800 rounded-3xl p-6 md:p-8 shadow-2xl border border-stone-700/50 flex flex-col items-center relative overflow-hidden">
          
          {/* Counter */}
          <div className="flex flex-col items-center justify-center mb-8 relative z-10">
            <div className="text-8xl md:text-[12rem] font-black tracking-tighter text-white tabular-nums leading-none">
              {count}
            </div>
            <div className="text-stone-400 font-medium uppercase tracking-widest mt-2 flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-500" />
              已完成
            </div>
          </div>

          {/* Status Indicator */}
          <div className={`px-6 py-2 rounded-full font-bold text-sm tracking-wide transition-colors duration-300 ${isBowed ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-stone-700/50 text-stone-300 border border-stone-600/50'}`}>
            {isBowed ? '跪拜中' : '直立'}
          </div>

          {/* Camera View */}
          <div className="w-full max-w-2xl aspect-video bg-stone-950 rounded-2xl overflow-hidden relative mt-8 border border-stone-700/50 shadow-inner">
            {!isReady && !error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-stone-400 space-y-4">
                <div className="w-8 h-8 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin"></div>
                <p>正在加载AI模型...</p>
              </div>
            )}
            
            {error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-red-400 p-6 text-center bg-red-950/20">
                <Info className="w-10 h-10 mb-2" />
                <p>{error}</p>
              </div>
            )}

            {!isRunning && isReady && !error && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-stone-400 bg-stone-900/80 backdrop-blur-sm z-20">
                <CameraOff className="w-12 h-12 mb-4 opacity-50" />
                <p>摄像头已关闭</p>
              </div>
            )}

            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-cover transform -scale-x-100"
              playsInline
              muted
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover transform -scale-x-100 z-10"
            />
          </div>

        </div>

        {/* Controls */}
        <div className="flex flex-col md:flex-row flex-wrap justify-center gap-4 items-center">
          
          {/* Camera Selector */}
          <div className="flex items-center gap-2 bg-stone-800 px-4 py-3 rounded-2xl border border-stone-700 w-full md:w-auto">
            <Settings className="w-5 h-5 text-stone-400" />
            <select 
              value={selectedDeviceId} 
              onChange={handleDeviceChange}
              className="bg-transparent text-stone-200 outline-none w-full md:w-48 text-sm"
            >
              {devices.length === 0 && <option value="">默认摄像头</option>}
              {devices.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId} className="bg-stone-800">
                  {device.label || `摄像头 ${index + 1}`}
                </option>
              ))}
            </select>
          </div>

          {!isRunning ? (
            <button
              onClick={startCamera}
              disabled={!isReady}
              className="flex-1 md:flex-none flex items-center justify-center gap-2 px-8 py-4 bg-emerald-600 hover:bg-emerald-500 disabled:bg-stone-700 disabled:text-stone-500 text-white rounded-2xl font-bold text-lg transition-all shadow-lg shadow-emerald-900/20 active:scale-95"
            >
              <Camera className="w-6 h-6" />
              开启摄像头
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="flex-1 md:flex-none flex items-center justify-center gap-2 px-8 py-4 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/20 rounded-2xl font-bold text-lg transition-all active:scale-95"
            >
              <CameraOff className="w-6 h-6" />
              关闭摄像头
            </button>
          )}

          <button
            onClick={resetCount}
            className="flex-1 md:flex-none flex items-center justify-center gap-2 px-8 py-4 bg-stone-800 hover:bg-stone-700 text-stone-300 rounded-2xl font-bold text-lg transition-all border border-stone-700 active:scale-95"
          >
            <RefreshCw className="w-6 h-6" />
            重置
          </button>
        </div>

        {/* Instructions */}
        <div className="bg-stone-800/50 rounded-2xl p-6 border border-stone-700/50 text-stone-400 text-sm leading-relaxed">
          <h3 className="text-stone-200 font-bold mb-2 flex items-center gap-2">
            <Info className="w-4 h-4" />
            使用说明
          </h3>
          <ul className="list-disc list-inside space-y-1 ml-1">
            <li>请将设备放置在能够清晰看到您全身或大半身的位置（侧面或正面均可）。</li>
            <li><strong>大拜判定标准：</strong>系统会自动学习您的动作幅度。请确保每次起身时完全直立，磕头时头部尽量贴近地面。</li>
            <li>中间的鞠躬和跪立动作不会被误判，只有完成“直立 -&gt; 磕头 -&gt; 直立”的完整周期才记为一次。</li>
            <li>如果发现有多个摄像头，可以在左下角的下拉菜单中进行切换。</li>
            <li>所有处理均在您的设备本地完成，不会录制视频或上传任何数据，保护您的隐私。</li>
          </ul>
        </div>

      </div>
    </div>
  );
}
