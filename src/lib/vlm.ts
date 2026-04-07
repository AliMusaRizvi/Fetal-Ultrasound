// VLM API Integration — Gradio (HuggingFace Spaces)
// Endpoint: https://alimusarizvi-fetal.hf.space
// Protocol: Gradio queue + SSE

const VLM_BASE = import.meta.env.VITE_VLM_API_URL || 'https://alimusarizvi-fetal.hf.space';

export interface PlaneResult {
  label: string;
  confidence: number;
  raw_text: string;
  all_probs?: Record<string, number>;
}

export interface BrainAnomalyResult {
  label: string;
  confidence: number;
  raw_text: string;
  is_normal: boolean;
}

export interface NTMarkerResult {
  nt_thickness_mm: number;
  nasal_bone_present: boolean;
  risk: 'Low' | 'High';
  raw_text: string;
  heatmap_url?: string;
}

export interface HeartSegmentationResult {
  ctr: number;
  cardiac_area?: number;
  thoracic_area?: number;
  interpretation: string;
  raw_text: string;
  overlay_url?: string;
}

export interface VLMPipelineResult {
  plane: PlaneResult;
  brain_anomaly?: BrainAnomalyResult;
  nt_markers?: NTMarkerResult;
  heart_segmentation?: HeartSegmentationResult;
  routed_to: 'brain' | 'nt' | 'heart' | 'other';
}

// ─── Upload image to Gradio then call an endpoint ─────────────────────────────

async function uploadToGradio(imageFile: File): Promise<string> {
  const form = new FormData();
  form.append('files', imageFile);
  const res = await fetch(`${VLM_BASE}/gradio_api/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(`Gradio upload failed: ${res.status}`);
  const paths: string[] = await res.json();
  return paths[0];
}

async function callGradioEndpoint(
  endpoint: string,
  uploadedPath: string
): Promise<{ text?: string; imageUrl?: string }> {
  // Step 1: Queue the prediction
  const queueRes = await fetch(`${VLM_BASE}/gradio_api/call/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: [{ path: uploadedPath, meta: { _type: 'gradio.FileData' } }],
    }),
  });
  if (!queueRes.ok) throw new Error(`Gradio call failed: ${queueRes.status}`);
  const { event_id } = await queueRes.json();

  // Step 2: Poll the SSE stream for the result
  return new Promise((resolve, reject) => {
    const eventSource = new EventSource(
      `${VLM_BASE}/gradio_api/call/${endpoint}/${event_id}`
    );

    eventSource.addEventListener('complete', (event: MessageEvent) => {
      eventSource.close();
      try {
        const parsed = JSON.parse(event.data);
        const outputs = parsed as Array<{ value?: string; url?: string; path?: string }>;
        let text: string | undefined;
        let imageUrl: string | undefined;
        for (const out of outputs) {
          if (typeof out === 'string') {
            text = out;
          } else if (out?.url) {
            imageUrl = out.url;
          } else if (out?.path) {
            imageUrl = `${VLM_BASE}/gradio_api/file=${out.path}`;
          }
        }
        resolve({ text, imageUrl });
      } catch {
        reject(new Error('Failed to parse Gradio response'));
      }
    });

    eventSource.addEventListener('error', () => {
      eventSource.close();
      reject(new Error('SSE stream error from Gradio'));
    });

    // Timeout after 90 seconds
    setTimeout(() => {
      eventSource.close();
      reject(new Error('Gradio request timed out'));
    }, 90_000);
  });
}

// ─── Individual Endpoint Wrappers ────────────────────────────────────────────

export async function classifyPlane(imageFile: File): Promise<PlaneResult> {
  const uploadedPath = await uploadToGradio(imageFile);
  const { text } = await callGradioEndpoint('predict_plane', uploadedPath);
  const raw = text || '';

  // Extract top label and confidence from markdown text
  const labelMatch = raw.match(/\*\*Plane:\s*([^*]+)\*\*/i) || raw.match(/Plane:\s*([^\n(]+)/i);
  const confMatch = raw.match(/confidence[:\s]+([0-9.]+)/i);
  let label = labelMatch ? labelMatch[1].trim() : 'unknown';
  
  // Cleanup any lingering markdown asterisks or colons
  label = label.replace(/[*:]/g, '').trim();
  
  if (label.toLowerCase() === 'unknown' && raw.length > 0) {
    label = raw.replace(/[*]/g, '').split('(')[0].trim() || 'unknown';
  }
  
  const confidence = confMatch ? parseFloat(confMatch[1]) : 0;

  return { label, confidence, raw_text: raw };
}

export async function detectBrainAnomaly(imageFile: File): Promise<BrainAnomalyResult> {
  const uploadedPath = await uploadToGradio(imageFile);
  const { text } = await callGradioEndpoint('predict_brain', uploadedPath);
  const raw = text || '';

  let label = 'unknown';
  
  // Try to match "Predicted condition:** Normal" or "Predicted condition: Normal" ignoring inner asterisks
  const predMatch = raw.match(/Predicted condition:?\*?\*?\s*([a-zA-Z\s-]+)/i);
  if (predMatch && predMatch[1] && predMatch[1].trim().length > 0) {
    label = predMatch[1].trim();
  } else {
    const classMatch = raw.match(/Class:\s*([^\n(]+)/i);
    if (classMatch) {
      label = classMatch[1].trim();
    } else {
      const boldMatch = raw.match(/\*\*([^*]+)\*\*/i);
      if (boldMatch && !boldMatch[1].toLowerCase().includes('condition')) {
        label = boldMatch[1].trim();
      }
    }
  }

  // Fallback cleanup
  if (label.toLowerCase() === 'unknown' || label === '') {
    if (raw.toLowerCase().includes('normal')) label = 'Normal';
    else label = raw.replace(/[*_]/g, '').split('(')[0].replace(/Predicted condition/i, '').replace(/:/g, '').trim() || 'unknown';
  }

  // Final scrub of asterisks
  label = label.replace(/[*]/g, '').trim();

  const confMatch = raw.match(/confidence[:\s]+([0-9.]+)/i);
  const confidence = confMatch ? parseFloat(confMatch[1]) : 0;
  
  // Always safely check the ENTIRE text for the word normal implicitly
  const is_normal = raw.toLowerCase().includes('normal');

  return { label, confidence, raw_text: raw, is_normal };
}

export async function detectNTMarkers(imageFile: File): Promise<NTMarkerResult> {
  const uploadedPath = await uploadToGradio(imageFile);
  const { text, imageUrl } = await callGradioEndpoint('predict_ds', uploadedPath);
  const raw = text || '';

  // Extract NT thickness
  const ntMatch = raw.match(/NT[:\s]+([0-9.]+)\s*mm/i) || raw.match(/([0-9.]+)\s*mm/i);
  const nt_thickness_mm = ntMatch ? parseFloat(ntMatch[1]) : 0;

  // Nasal bone
  const nasalMatch = raw.match(/nasal bone[:\s]*(present|absent|yes|no)/i);
  const nasal_bone_present = nasalMatch
    ? ['present', 'yes'].includes(nasalMatch[1].toLowerCase())
    : true;

  // Risk
  const risk: 'Low' | 'High' = nt_thickness_mm >= 3.5 ? 'High' : 'Low';

  return { nt_thickness_mm, nasal_bone_present, risk, raw_text: raw, heatmap_url: imageUrl };
}

export async function segmentHeart(imageFile: File): Promise<HeartSegmentationResult> {
  const uploadedPath = await uploadToGradio(imageFile);
  const { text, imageUrl } = await callGradioEndpoint('predict_segmentation', uploadedPath);
  const raw = text || '';

  const ctrMatch = raw.match(/CTR[:\s]+([0-9.]+)/i);
  const cardiacMatch = raw.match(/cardiac area[:\s]+([0-9.]+)/i);
  const thoracicMatch = raw.match(/thoracic area[:\s]+([0-9.]+)/i);

  const ctr = ctrMatch ? parseFloat(ctrMatch[1]) : 0;
  const cardiac_area = cardiacMatch ? parseFloat(cardiacMatch[1]) : undefined;
  const thoracic_area = thoracicMatch ? parseFloat(thoracicMatch[1]) : undefined;
  const interpretation = ctr > 0.55
    ? 'Cardiomegaly suspected (CTR > 0.55). Recommend fetal echocardiogram.'
    : 'CTR within normal range.';

  return { ctr, cardiac_area, thoracic_area, interpretation, raw_text: raw, overlay_url: imageUrl };
}

// ─── Main Pipeline — routes based on plane classification ─────────────────────

export async function runVLMPipeline(imageFile: File): Promise<VLMPipelineResult> {
  // Step 1: Classify plane
  const plane = await classifyPlane(imageFile);
  const planeLabel = plane.label.toLowerCase();
  const rawText = plane.raw_text.toLowerCase();

  // Step 2: Route to appropriate head
  const brainPlanes = [
    'brain-thalamic', 'brain thalamic', 
    'brain-cerebellum', 'brain cerebellum', 
    'brain verticular', 'brain-ventricular', 'brain ventricular',
    'brain-tv', 'brain tv', 
    'brain-cb', 'brain cb', 
    'transventricular', 'thalamic', 'cerebellum'
  ];

  if (brainPlanes.some(bp => planeLabel.includes(bp) || rawText.includes(bp))) {
    const brain_anomaly = await detectBrainAnomaly(imageFile);
    return { plane, brain_anomaly, routed_to: 'brain' };
  }

  if (planeLabel.includes('sagg') || planeLabel.includes('sagittal') || planeLabel.includes('mid-sag') || planeLabel.includes('profile')) {
    const nt_markers = await detectNTMarkers(imageFile);
    return { plane, nt_markers, routed_to: 'nt' };
  }

  if (planeLabel.includes('thorax') || planeLabel.includes('4ch') || planeLabel.includes('four') || planeLabel.includes('cardiac') || planeLabel.includes('heart')) {
    const heart_segmentation = await segmentHeart(imageFile);
    return { plane, heart_segmentation, routed_to: 'heart' };
  }

  // Other planes (femur, abdominal, etc.) — return plane result only
  return { plane, routed_to: 'other' };
}
