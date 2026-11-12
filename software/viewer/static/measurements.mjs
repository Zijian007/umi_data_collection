import {nearestIndex} from './math.mjs';

function formatTagDistance(measurement, frameIndex) {
  const distanceMm = measurement?.distance_mm[frameIndex];
  return measurement?.valid[frameIndex] && Number.isFinite(distanceMm) ? distanceMm.toFixed(2) : '—';
}

export function tagDistanceReadout(recording, frameIndex) {
  const hasMeasurement = Boolean(recording.tag_distance || recording.companion?.tag_distance);
  const primaryDistance = formatTagDistance(recording.tag_distance, frameIndex);
  if (!recording.companion) return {isVisible: hasMeasurement, text: primaryDistance};

  const companion = recording.companion;
  const companionIndex = nearestIndex(companion.times, recording.times[frameIndex]);
  const primaryLabel = recording.primary_arm === 'right' ? '右' : '左';
  const companionLabel = companion.arm === 'right' ? '右' : '左';
  const companionDistance = formatTagDistance(companion.tag_distance, companionIndex);
  return {isVisible: hasMeasurement, text: `${primaryLabel} ${primaryDistance} / ${companionLabel} ${companionDistance}`};
}
