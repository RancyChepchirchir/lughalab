type Point = {
  x: number;
  y: number;
  z: number;
};

type LandmarkSkeletonProps = {
  landmarks: number[];
  width?: number;
  height?: number;
};

const POSE_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],

  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],

  [9, 10],

  [11, 12],

  [11, 13],
  [13, 15],

  [12, 14],
  [14, 16],

  [11, 23],
  [12, 24],
  [23, 24],

  [23, 25],
  [25, 27],

  [24, 26],
  [26, 28],

  [27, 29],
  [29, 31],

  [28, 30],
  [30, 32],
];

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],

  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],

  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],

  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],

  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],

  [0, 17],
];

function unpack(
  values: number[],
  start: number,
  count: number
): Point[] {
  const points: Point[] = [];

  for (let index = 0; index < count; index++) {
    const offset = start + index * 3;

    points.push({
      x: values[offset] ?? 0,
      y: values[offset + 1] ?? 0,
      z: values[offset + 2] ?? 0,
    });
  }

  return points;
}

function valid(point: Point) {
  return !(
    point.x === 0 &&
    point.y === 0 &&
    point.z === 0
  );
}

export default function LandmarkSkeleton({
  landmarks,
  width = 720,
  height = 500,
}: LandmarkSkeletonProps) {
  /*
   * Current LughaLab representation:
   *
   * pose:
   *   33 × xyz = 99
   *
   * left hand:
   *   21 × xyz = 63
   *
   * right hand:
   *   21 × xyz = 63
   *
   * total:
   *   225
   */

  const pose = unpack(landmarks, 0, 33);
  const leftHand = unpack(landmarks, 99, 21);
  const rightHand = unpack(landmarks, 162, 21);

  function sx(point: Point) {
    return point.x * width;
  }

  function sy(point: Point) {
    return point.y * height;
  }

  function renderConnections(
    points: Point[],
    connections: [number, number][],
    className: string
  ) {
    return connections.map(([a, b], index) => {
      const p1 = points[a];
      const p2 = points[b];

      if (!p1 || !p2 || !valid(p1) || !valid(p2)) {
        return null;
      }

      return (
        <line
          key={`${a}-${b}-${index}`}
          x1={sx(p1)}
          y1={sy(p1)}
          x2={sx(p2)}
          y2={sy(p2)}
          className={className}
          vectorEffect="non-scaling-stroke"
        />
      );
    });
  }

  function renderPoints(
    points: Point[],
    className: string,
    radius: number
  ) {
    return points.map((point, index) => {
      if (!valid(point)) return null;

      return (
        <circle
          key={index}
          cx={sx(point)}
          cy={sy(point)}
          r={radius}
          className={className}
          vectorEffect="non-scaling-stroke"
        />
      );
    });
  }

  const poseDetected = pose.filter(valid).length;
  const leftDetected = leftHand.filter(valid).length;
  const rightDetected = rightHand.filter(valid).length;

  return (
    <div className="relative overflow-hidden border border-white/10 bg-[#06080a]">
      <div className="absolute left-4 top-4 z-10 font-mono text-[9px] uppercase tracking-[0.16em] text-white/20">
        Normalised image coordinates
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="aspect-[720/500] w-full"
        role="img"
        aria-label="Pose and hand landmark skeleton"
      >
        <defs>
          <pattern
            id="landmark-grid"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
              className="text-white/[0.035]"
            />
          </pattern>
        </defs>

        <rect
          width={width}
          height={height}
          fill="url(#landmark-grid)"
        />

        <line
          x1={width / 2}
          y1="0"
          x2={width / 2}
          y2={height}
          stroke="currentColor"
          strokeWidth="0.5"
          className="text-white/[0.06]"
        />

        <line
          x1="0"
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="currentColor"
          strokeWidth="0.5"
          className="text-white/[0.06]"
        />

        <g>
          {renderConnections(
            pose,
            POSE_CONNECTIONS,
            "stroke-white/25"
          )}

          {renderPoints(
            pose,
            "fill-white/45",
            2.6
          )}
        </g>

        <g>
          {renderConnections(
            leftHand,
            HAND_CONNECTIONS,
            "stroke-emerald-300/55"
          )}

          {renderPoints(
            leftHand,
            "fill-emerald-300/80",
            3
          )}
        </g>

        <g>
          {renderConnections(
            rightHand,
            HAND_CONNECTIONS,
            "stroke-cyan-300/55"
          )}

          {renderPoints(
            rightHand,
            "fill-cyan-300/80",
            3
          )}
        </g>
      </svg>

      <div className="grid grid-cols-3 border-t border-white/10">
        <Legend
          label="Pose"
          value={`${poseDetected}/33`}
          dotClass="bg-white/50"
        />

        <Legend
          label="Left hand"
          value={`${leftDetected}/21`}
          dotClass="bg-emerald-300/80"
        />

        <Legend
          label="Right hand"
          value={`${rightDetected}/21`}
          dotClass="bg-cyan-300/80"
        />
      </div>
    </div>
  );
}

function Legend({
  label,
  value,
  dotClass,
}: {
  label: string;
  value: string;
  dotClass: string;
}) {
  return (
    <div className="border-r border-white/10 p-3 last:border-r-0">
      <div className="flex items-center gap-2">
        <span
          className={`h-1.5 w-1.5 rounded-full ${dotClass}`}
        />

        <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-white/25">
          {label}
        </span>
      </div>

      <p className="mt-2 font-mono text-xs text-white/55">
        {value}
      </p>
    </div>
  );
}