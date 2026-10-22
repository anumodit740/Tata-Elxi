import React, { useState, useId, useRef, useEffect } from 'react';
import { createRoot } from 'react-dom/client';

const ink = '#142d3c';
const teal = '#087e83';
const amber = '#a46c16';
const red = '#bd4c4b';
const number = (value, digits = 1) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
const integer = (value) => Number.isFinite(Number(value)) ? Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 }) : '—';
const titleCase = (value) => String(value || '').replace(/_/g, ' ').toLowerCase().replace(/^\w/, (c) => c.toUpperCase());

function useChartWidth(initialWidth = 600) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initialWidth);
  useEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    const measure = (value) => {
      const next = Math.max(280, Math.floor(value));
      if (Number.isFinite(next)) setWidth((previous) => previous === next ? previous : next);
    };
    measure(element.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver((entries) => {
      const entry = entries.find((item) => item.target === element);
      if (entry) measure(entry.contentRect.width);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

function Icon({ name = 'pulse', size = 18, ...rest }) {
  const paths = {
    pulse: <><path d="M3 12h4l2-6 4 12 2-6h6" /></>,
    engine: <><path d="M6 7h11l4 4v7H6l-3-3V9h3Z" /><path d="M9 7V4h6v3M6 11H3m14 0h4M9 11v4m4-4v4" /></>,
    shield: <><path d="m12 3 8 3v5c0 5-8 10-8 10S4 16 4 11V6Z" /><path d="m8 12 3 3 5-6" /></>,
    arrow: <><path d="M4 12h15m-5-5 5 5-5 5" /></>,
    chart: <><path d="M4 4v16h16M7 14l4-5 4 3 5-7" /></>,
    clipboard: <><rect x="6" y="5" width="13" height="16" rx="2" /><path d="M9 5V3h7v2M10 10h5m-5 4h5m-5 4h3" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.2" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    layers: <><path d="m12 3 10 6-10 6L2 9Zm-9 11 9 5 9-5M3 18l9 5 9-5" /></>,
    table: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 10h18M9 4v16M3 15h18" /></>,
    search: <><circle cx="10" cy="10" r="6" /><path d="m15 15 6 6" /></>,
    chevron: <path d="m9 5 7 7-7 7" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...rest}>{paths[name] || paths.pulse}</svg>;
}

function Severity({ value, prefix }) {
  const severity = String(value || 'UNAVAILABLE').toUpperCase();
  const tone = ['CRITICAL', 'WARNING', 'WATCH', 'NORMAL'].includes(severity) ? severity.toLowerCase() : 'neutral';
  return <span className={`af-severity af-severity--${tone}`}>{prefix && <span>{prefix}</span>}<i />{severity}</span>;
}

function Eyebrow({ icon, children }) {
  return <div className="af-eyebrow">{icon && <Icon name={icon} size={15} />}{children}</div>;
}

function EmptyState({ data }) {
  const steps = [
    { icon: 'shield', title: 'Validate', detail: 'Complete, ordered sensor history' },
    { icon: 'chart', title: 'Predict', detail: 'Frozen model, fresh inference' },
    { icon: 'pulse', title: 'Triage', detail: 'Confirmed RUL thresholds' },
    { icon: 'clipboard', title: 'Work order', detail: 'Persisted, idempotent action' },
  ];
  return <div className="af-empty">
    <div className="af-empty-copy">
      <Eyebrow icon="engine">ENGINE DECISION WORKSPACE</Eyebrow>
      {data.message ? <h2>{data.message}</h2> : <h2>See the signal.<br /><span>Understand the action.</span></h2>}
      <p>{data.detail || 'Select an available history and observed cycle above, then run the model and agent. Predictions and maintenance evidence will appear together.'}</p>
      <div className="af-context-row"><span>{data.subset || 'C-MAPSS'}</span>{data.engine != null && <span>Engine {integer(data.engine)}</span>}<span>{data.source_label || 'Waiting for input'}</span></div>
    </div>
    <div className="af-empty-process" aria-label="Processing flow: Validate, Predict, Triage, Work order">
      {steps.map((step, index) => <React.Fragment key={step.title}><div className="af-process-step"><div className="af-process-icon"><Icon name={step.icon} size={22} /></div><div><strong>{step.title}</strong><span>{step.detail}</span></div></div>{index < steps.length - 1 && <div className="af-process-connector"><Icon name="arrow" size={16} /></div>}</React.Fragment>)}
    </div>
    <div className="af-empty-foot"><Icon name="info" size={16} /><p>No prediction is displayed before a successful run. NASA example histories are fitting-TRAIN data; this workspace does not estimate generalization.</p></div>
  </div>;
}

function ActualTable({ rows, fields }) {
  return <div className="af-table-scroll" role="region" aria-label="Actual chart observations" tabIndex={0}><table className="af-data-table"><thead><tr>{fields.map((field) => <th key={field.key} scope="col">{field.label}</th>)}</tr></thead><tbody>{rows.map((row) => <tr key={`${row.engine_id}-${row.cycle}`}>{fields.map((field) => <td key={field.key}>{field.key === 'cycle' ? integer(row[field.key]) : number(row[field.key], field.digits ?? 3)}</td>)}</tr>)}</tbody></table></div>;
}

function RulChart({ rows }) {
  const [window, setWindow] = useState('all');
  const [activeCycle, setActiveCycle] = useState(null);
  const [table, setTable] = useState(false);
  const descriptionId = useId();
  const [chartRef, width] = useChartWidth();
  const lastCycle = Number(rows[rows.length - 1]?.cycle || 0);
  const visible = window === 'recent' ? rows.filter((row) => Number(row.cycle) >= lastCycle - 59) : rows;
  if (!visible.length) return <div className="af-no-data">No valid RUL observations are available.</div>;
  const height = 300, pad = { l: 45, r: 17, t: 23, b: 45 };
  const plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
  const minX = Number(visible[0].cycle), maxX = Number(visible[visible.length - 1].cycle);
  const maxY = Math.max(90, Math.ceil(Math.max(...visible.map((r) => Number(r.predicted_rul))) / 30) * 30);
  const x = (cycle) => maxX === minX ? pad.l + plotW / 2 : pad.l + (Number(cycle) - minX) / (maxX - minX) * plotW;
  const y = (value) => pad.t + plotH - Number(value) / maxY * plotH;
  const active = visible.find((row) => Number(row.cycle) === activeCycle) || visible[visible.length - 1];
  const path = visible.map((row, index) => `${index ? 'L' : 'M'}${x(row.cycle).toFixed(2)},${y(row.predicted_rul).toFixed(2)}`).join(' ');
  const yTicks = Array.from(new Set([0, 30, 60, maxY]));
  const xTicks = Array.from(new Set([minX, Math.round((minX + maxX) / 2), maxX]));
  const moveAt = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const chartX = (event.clientX - rect.left) / rect.width * width;
    const target = minX + (chartX - pad.l) / plotW * (maxX - minX);
    const nearest = visible.reduce((best, row) => Math.abs(Number(row.cycle) - target) < Math.abs(Number(best.cycle) - target) ? row : best, visible[0]);
    setActiveCycle(Number(nearest.cycle));
  };
  const keyMove = (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    let index = visible.findIndex((row) => Number(row.cycle) === Number(active.cycle));
    if (event.key === 'Home') index = 0;
    if (event.key === 'End') index = visible.length - 1;
    if (event.key === 'ArrowLeft') index = Math.max(0, index - 1);
    if (event.key === 'ArrowRight') index = Math.min(visible.length - 1, index + 1);
    setActiveCycle(Number(visible[index].cycle));
  };
  return <>
    <div className="af-chart-toolbar"><div className="af-chart-legend"><span><i className="af-line-key" />Predicted RUL</span><span><i className="af-dot-key af-dot-key--amber" />Warning band</span><span><i className="af-dot-key af-dot-key--red" />Critical band</span></div><div className="af-segment" aria-label="Chart history window"><button aria-pressed={window === 'all'} onClick={() => setWindow('all')}>Full history</button><button aria-pressed={window === 'recent'} onClick={() => setWindow('recent')}>Last 60 cycles</button></div></div>
    <div className="af-chart-readout" aria-live="polite"><span>Observed cycle <b>{integer(active.cycle)}</b></span><span>Predicted RUL <b>{number(active.predicted_rul)} <small>cycles</small></b></span></div>
    <svg ref={chartRef} className="af-rul-svg" style={{ height }} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Predicted remaining useful life from cycle ${minX} to ${maxX}. Use left and right arrow keys to inspect observations.`} aria-describedby={descriptionId} tabIndex={0} onPointerMove={moveAt} onKeyDown={keyMove}>
      <title>Predicted remaining useful life, actual model outputs</title>
      <rect x={pad.l} y={pad.t} width={plotW} height={plotH} fill="#f7fbfb" />
      <rect x={pad.l} y={y(60)} width={plotW} height={y(30) - y(60)} fill="#fcf5e7" />
      <rect x={pad.l} y={y(30)} width={plotW} height={y(0) - y(30)} fill="#fbefed" />
      {yTicks.map((value) => <g key={value}><line x1={pad.l} x2={width - pad.r} y1={y(value)} y2={y(value)} stroke={value === 30 ? '#d99d97' : value === 60 ? '#d9b66d' : '#dfe8eb'} strokeDasharray={value === 30 || value === 60 ? '5 5' : undefined} /><text x={pad.l - 13} y={y(value) + 4} textAnchor="end" className="af-svg-tick">{integer(value)}</text></g>)}
      {xTicks.map((value) => <text key={value} x={x(value)} y={height - 20} textAnchor="middle" className="af-svg-tick">{integer(value)}</text>)}
      <text x={pad.l} y={13} className="af-svg-axis">RUL · cycles</text><text x={width - pad.r} y={height - 3} textAnchor="end" className="af-svg-axis">Observed cycle</text>
      <path d={path} stroke={teal} strokeWidth="2.7" fill="none" strokeLinejoin="round" strokeLinecap="round" />
      <line x1={x(active.cycle)} x2={x(active.cycle)} y1={pad.t} y2={pad.t + plotH} stroke="#65969a" strokeDasharray="3 4" />
      <circle cx={x(active.cycle)} cy={y(active.predicted_rul)} r="5" fill={teal} stroke="white" strokeWidth="2.5" />
      <g transform={`translate(${Math.max(pad.l + 3, Math.min(width - pad.r - 156, x(active.cycle) - 77))},${Math.max(pad.t + 3, y(active.predicted_rul) - 62)})`} pointerEvents="none"><rect width="153" height="47" rx="6" fill="white" stroke="#ccdedf" /><text x="11" y="18" className="af-svg-tooltip-label">Cycle {integer(active.cycle)}</text><text x="11" y="35" className="af-svg-tooltip-value">RUL {number(active.predicted_rul)} cycles</text></g>
    </svg>
    <div className="af-chart-bottom"><p id={descriptionId}>Hover or focus the chart and use arrow keys. Every point is a model output; shaded bands are policy thresholds.</p><button className="af-text-button" onClick={() => setTable(!table)} aria-expanded={table}><Icon name="table" size={15} />{table ? 'Hide observations' : 'View observations'}</button></div>
    {table && <ActualTable rows={visible} fields={[{ key: 'cycle', label: 'Observed cycle' }, { key: 'predicted_rul', label: 'Predicted RUL · cycles' }]} />}
  </>;
}

function PolicyPanel({ data }) {
  const open = data.orders.filter((order) => order.status === 'OPEN');
  const severity = data.summary.severity;
  const decision = data.decisions?.[data.decisions.length - 1]?.decision;
  return <aside className="af-card af-policy-card">
    <Eyebrow icon="shield">FROZEN TRIAGE POLICY</Eyebrow><h3>From prediction<br />to action</h3>
    <div className="af-policy-status"><span>Active display severity</span><Severity value={severity} /></div>
    <div className="af-threshold-list">
      <div><span className="af-threshold-dot af-threshold-dot--red" /><strong>≤5 <small>cycles</small></strong><span>Emergency<br /><b>Immediate action</b></span></div>
      <div><span className="af-threshold-dot af-threshold-dot--red" /><strong>≤30 <small>cycles</small></strong><span>Critical<br /><b>Confirm consecutive cycles</b></span></div>
      <div><span className="af-threshold-dot af-threshold-dot--amber" /><strong>≤60 <small>cycles</small></strong><span>Warning<br /><b>Confirm consecutive cycles</b></span></div>
    </div>
    <p className="af-policy-note">Confirmation: 2 consecutive cycles. Display clearance: 10-cycle hysteresis and 3 clear cycles. An open order requires manual closure.</p>
    <div className="af-order-link"><span>Open-order severity</span>{open.length ? open.map((order) => <Severity key={order.work_order_id} value={order.severity} />) : <strong>No open order</strong>}</div>
    {decision?.reason_codes?.length > 0 && <div className="af-live-reasons"><span>Latest decision</span>{decision.reason_codes.map((reason) => <p key={reason}>{reasonText(reason)}</p>)}</div>}
    <div className="af-policy-scope"><Icon name="info" size={15} /><span>Demonstration rules. A model threshold is not a diagnosis or maintenance approval.</span></div>
  </aside>;
}

const reasons = {
  RUL_AT_OR_BELOW_CRITICAL: 'Predicted RUL reached the critical threshold.',
  RUL_AT_OR_BELOW_WARNING: 'Predicted RUL reached the warning threshold.',
  EMERGENCY_RUL_IMMEDIATE: 'Emergency RUL permits immediate action.',
  CONFIRMED_ACTIONABLE: 'Consecutive-cycle confirmation met.',
  IMMEDIATE_ACTIONABLE: 'Emergency action confirmed immediately.',
  AWAITING_CONSECUTIVE_CONFIRMATION: 'Waiting for consecutive-cycle confirmation.',
  HELD_BY_HYSTERESIS: 'Display severity retained by hysteresis.',
  DISPLAY_SEVERITY_CLEARED_AFTER_HYSTERESIS: 'Display severity cleared after sustained recovery.',
  EXISTING_ORDER_REQUIRES_MANUAL_CLOSURE: 'The existing order still requires manual closure.',
  MAD_NOVELTY_SECONDARY_ONLY: 'Robust sensor deviation is informational only.',
  IDENTICAL_EVENT: 'Identical observation replayed without another action.',
  NO_THRESHOLD_TRIGGER: 'No confirmed maintenance threshold triggered.',
};
function reasonText(reason) { return reasons[reason] || titleCase(reason); }

function RulAnalysis({ data }) {
  return <div className="af-analysis-grid"><section className="af-card af-main-chart"><div className="af-card-head"><div><Eyebrow icon="chart">CAUSAL PREDICTION HISTORY</Eyebrow><h3>Remaining useful life</h3><p>Sensor history through cycle {integer(data.observed_cycle)} only.</p></div><span className="af-unit-badge">OPERATING CYCLES</span></div><RulChart rows={data.rows} /></section><PolicyPanel data={data} /><section className="af-interpretation"><Icon name="info" size={18} /><div><strong>Read the model and the order separately.</strong><p>RUL estimates remaining operating cycles. Confirmation and hysteresis can make the active severity differ from a single prediction; an open maintenance order remains open until a human closes it.</p></div></section></div>;
}

function EvidencePair({ label, value, mono = false }) {
  return <div className="af-evidence-pair"><dt>{label}</dt><dd className={mono ? 'af-mono' : ''}>{value ?? 'Not recorded'}</dd></div>;
}

function OrderCard({ order }) {
  const evidence = order.latest_evidence || order.evidence || {};
  const history = Array.isArray(order.history) ? order.history : [];
  return <article className="af-card af-order-card"><div className="af-order-head"><div className="af-order-identity"><div className="af-order-icon"><Icon name="clipboard" size={24} /></div><div><Eyebrow>{order.asset?.subset} · ENGINE {integer(order.asset?.engine_id)}</Eyebrow><h3>{titleCase(order.action)} order</h3><code>{order.work_order_id}</code></div></div><div className="af-order-badges"><span className={`af-state-pill ${order.status === 'OPEN' ? 'is-open' : ''}`}>{order.status}</span><Severity value={order.severity} /></div></div>
    <dl className="af-order-facts"><EvidencePair label="Created at cycle" value={integer(order.created_cycle)} /><EvidencePair label="Last observed cycle" value={integer(order.last_observed_cycle)} /><EvidencePair label="Episode" value={integer(order.episode)} /><EvidencePair label="Model version" value={order.model_version} /></dl>
    <div className="af-order-body"><div><div className="af-section-label"><Icon name="clock" size={15} />RECORDED ORDER HISTORY</div><ol className="af-history">{history.map((event, index) => <li key={`${event.cycle}-${index}`}><span className="af-history-mark" /><div><div className="af-history-title"><strong>Cycle {integer(event.cycle)}</strong><Severity value={event.severity} /></div><ul>{(event.reason_codes || []).map((reason) => <li key={reason}>{reasonText(reason)}</li>)}</ul></div></li>)}</ol></div><aside className="af-order-evidence"><div className="af-section-label"><Icon name="search" size={15} />LATEST RECORDED EVIDENCE</div><dl><EvidencePair label="Predicted RUL" value={`${number(evidence.predicted_rul)} cycles`} /><EvidencePair label="Event" value={evidence.event_id} mono /><EvidencePair label="Model version" value={evidence.model_version} /><EvidencePair label="MAD role" value="Secondary novelty only" /></dl><p>Classifier controls in order evidence are disabled by the frozen RUL-only policy. Actual classifier outputs appear under Signal diagnostics.</p></aside></div>
    <details className="af-hash-details"><summary>Inspect input fingerprint and policy scope</summary><dl><EvidencePair label="Input SHA-256" value={evidence.input_hash} mono /><EvidencePair label="Policy scope" value={order.policy_scope} mono /></dl></details>
    <div className="af-order-footer"><Icon name="info" size={15} /><span>{order.status === 'OPEN' ? 'This order remains open even if the predicted RUL recovers. Closure is a separate human action.' : `Closure: ${order.closure_reason || 'No reason recorded'}`}</span></div>
  </article>;
}

function Orders({ data }) {
  return <div className="af-orders"><div className="af-section-intro"><div><Eyebrow icon="clipboard">AUDITABLE MAINTENANCE</Eyebrow><h3>Persisted work orders</h3><p>Order episodes and escalation history from this session, not simulated notifications.</p></div><span className="af-small-counter">{integer(data.orders.length)} recorded</span></div>{data.orders.length ? data.orders.map((order) => <OrderCard key={order.work_order_id} order={order} />) : <div className="af-card af-no-orders"><div className="af-no-orders-icon"><Icon name="shield" size={30} /></div><h3>No confirmed work order</h3><p>No confirmed maintenance threshold created an order in the selected history. This does not establish that the engine is fault-free.</p><div className="af-current-state"><span>Active display severity</span><Severity value={data.summary.severity} /></div></div>}<div className="af-inline-notice"><Icon name="info" size={16} /><p>Session storage is temporary. Use the download controls below the workspace to preserve orders and the inference receipt.</p></div></div>;
}

function MiniChart({ rows, field, thresholdField, color, label, valueLabel }) {
  const [table, setTable] = useState(false);
  const [chartRef, w] = useChartWidth();
  if (!rows.length) return null;
  const h = 170, p = { l: 42, r: 17, t: 16, b: 29 };
  const lo = Number(rows[0].cycle), hi = Number(rows[rows.length - 1].cycle);
  const limit = Number(rows[rows.length - 1][thresholdField]);
  const maxY = field === 'warning_probability' ? 1 : Math.max(1, limit * 1.1, ...rows.map((r) => Number(r[field]) * 1.05));
  const x = (v) => hi === lo ? (w + p.l - p.r) / 2 : p.l + (Number(v) - lo) / (hi - lo) * (w - p.l - p.r);
  const y = (v) => p.t + (1 - Number(v) / maxY) * (h - p.t - p.b);
  const path = rows.map((row, index) => `${index ? 'L' : 'M'}${x(row.cycle)},${y(row[field])}`).join(' ');
  return <><svg ref={chartRef} className="af-mini-svg" style={{ height: h }} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${label} over observed cycles ${lo} to ${hi}; dashed line is the frozen threshold ${number(limit, 3)}`}><title>{label}: actual model observations</title>{[0, maxY].map((tick) => <g key={tick}><line x1={p.l} x2={w - p.r} y1={y(tick)} y2={y(tick)} stroke="#e3eaed" /><text x={p.l - 8} y={y(tick) + 4} textAnchor="end" className="af-svg-tick">{number(tick, field === 'warning_probability' ? 1 : 0)}</text></g>)}<line x1={p.l} x2={w - p.r} y1={y(limit)} y2={y(limit)} stroke="#b39458" strokeDasharray="5 4" /><path d={path} fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" /><circle cx={x(hi)} cy={y(rows[rows.length - 1][field])} r="3.5" fill={color} />{Array.from(new Set([lo, hi])).map((tick) => <text key={tick} x={x(tick)} y={h - 8} textAnchor="middle" className="af-svg-tick">{integer(tick)}</text>)}</svg><div className="af-signal-chart-foot"><span>Observed cycle</span><button className="af-text-button" onClick={() => setTable(!table)} aria-expanded={table}><Icon name="table" size={14} />{table ? 'Hide observations' : 'View observations'}</button></div>{table && <ActualTable rows={rows} fields={[{ key: 'cycle', label: 'Cycle' }, { key: field, label: valueLabel }, { key: thresholdField, label: 'Frozen threshold' }]} />}</>;
}

function Diagnostics({ data }) {
  const last = data.rows[data.rows.length - 1];
  const horizon = last.warning_horizon;
  const classifierAbove = Number(last.warning_probability) >= Number(last.warning_threshold);
  const madAbove = Number(last.mad_score) > Number(last.mad_threshold);
  return <div className="af-diagnostics"><div className="af-section-intro"><div><Eyebrow icon="pulse">INFORMATIONAL SIGNALS</Eyebrow><h3>Abnormal-behaviour diagnostics</h3><p>These signals are visible for inspection. The frozen RUL-only policy does not use them to create or escalate an order.</p></div><span className="af-neutral-pill">NO AGENT ACTION</span></div><div className="af-diagnostics-grid"><article className="af-card af-signal-card"><div className="af-signal-heading"><div><h3>Near-failure proxy</h3><p>Estimated probability of true RUL ≤{integer(horizon)} cycles.</p></div><span className="af-signal-role">PROXY LABEL</span></div><div className="af-signal-value"><strong>{number(last.warning_probability, 3)}</strong><span>Frozen threshold <b>{number(last.warning_threshold, 3)}</b></span></div><MiniChart rows={data.rows} field="warning_probability" thresholdField="warning_threshold" color={teal} label="Near-failure proxy probability" valueLabel="Proxy probability" /><div className="af-signal-reading"><span className={`af-reading-dot ${classifierAbove ? 'is-raised' : ''}`} /><p>Latest signal is {classifierAbove ? 'at or above' : 'below'} its frozen threshold. This is a RUL-horizon proxy, not a physical fault-onset label.</p></div></article><article className="af-card af-signal-card"><div className="af-signal-heading"><div><h3>Robust sensor deviation</h3><p>MAD score relative to the fitted reference.</p></div><span className="af-signal-role">NOVELTY SIGNAL</span></div><div className="af-signal-value"><strong>{number(last.mad_score, 2)}</strong><span>Frozen threshold <b>{number(last.mad_threshold, 2)}</b></span></div><MiniChart rows={data.rows} field="mad_score" thresholdField="mad_threshold" color="#668395" label="MAD robust deviation score" valueLabel="MAD score" /><div className="af-signal-reading"><span className={`af-reading-dot ${madAbove ? 'is-raised' : ''}`} /><p>Latest score is {madAbove ? 'above' : 'at or below'} its frozen threshold. A deviation is not a calibrated failure probability.</p></div></article></div><div className="af-inline-notice"><Icon name="info" size={16} /><p>Do not read these panels as uncertainty intervals, fault diagnosis or added maintenance benefit. RUL confirmation and hysteresis drive the recorded actions.</p></div></div>;
}

function Dashboard({ data }) {
  const [tab, setTab] = useState('analysis');
  const tabPrefix = useId();
  if (!data || data.mode !== 'ready') return <div className="af-dashboard"><EmptyState data={data || {}} /></div>;
  const rows = Array.isArray(data.rows) ? data.rows.filter((row) => Number.isFinite(Number(row.cycle)) && Number.isFinite(Number(row.predicted_rul))) : [];
  const normalized = { ...data, rows, orders: Array.isArray(data.orders) ? data.orders : [], summary: data.summary || {} };
  if (!rows.length) return <div className="af-dashboard"><EmptyState data={normalized} /></div>;
  const tabs = [{ id: 'analysis', name: 'RUL analysis', icon: 'chart' }, { id: 'orders', name: 'Maintenance orders', icon: 'clipboard', count: normalized.orders.length }, { id: 'signals', name: 'Signal diagnostics', icon: 'pulse' }];
  return <div className="af-dashboard"><div className="af-workspace-top"><div className="af-workspace-context"><Icon name="engine" size={17} /><strong>{data.subset}</strong><span>/</span><span>Engine {integer(data.engine)}</span><span>/</span><span>Through cycle {integer(data.observed_cycle)}</span></div><span className="af-verified-label"><Icon name="check" size={14} />FROZEN MODEL · FRESH INFERENCE</span></div><nav className="af-tabs" role="tablist" aria-label="Engine workspace views">{tabs.map((item, index) => <button key={item.id} id={`${tabPrefix}-${item.id}`} role="tab" aria-selected={tab === item.id} aria-controls={`${tabPrefix}-panel`} tabIndex={tab === item.id ? 0 : -1} onClick={() => setTab(item.id)} onKeyDown={(event) => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) { event.preventDefault(); let next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length; setTab(tabs[next].id); event.currentTarget.parentElement.querySelectorAll('[role="tab"]')[next].focus(); } }}><Icon name={item.icon} size={17} />{item.name}{item.count !== undefined && <span className="af-tab-count">{integer(item.count)}</span>}</button>)}</nav><section id={`${tabPrefix}-panel`} role="tabpanel" aria-labelledby={`${tabPrefix}-${tab}`} className="af-tab-panel">{tab === 'analysis' && <RulAnalysis data={normalized} />}{tab === 'orders' && <Orders data={normalized} />}{tab === 'signals' && <Diagnostics data={normalized} />}</section></div>;
}

export default function mount({ data, parentElement }) {
  // Streamlit may call this renderer again for the same parent before invoking
  // its previous cleanup. Keep one owned host and root, then render new data.
  // The generation guard makes a delayed, older cleanup harmless.
  const hosts = Array.from(parentElement.children).filter((child) => child.classList.contains('af-react-host'));
  let host = hosts.find((child) => child.__aiFbiDashboardMount);
  if (!host) {
    host = document.createElement('div');
    host.className = 'af-react-host';
    parentElement.appendChild(host);
  }
  for (const duplicate of hosts) {
    if (duplicate === host) continue;
    const stale = duplicate.__aiFbiDashboardMount;
    if (stale) stale.root.unmount();
    duplicate.remove();
  }
  let instance = host.__aiFbiDashboardMount;
  if (!instance) {
    instance = { root: createRoot(host), generation: 0 };
    host.__aiFbiDashboardMount = instance;
  }
  const generation = ++instance.generation;
  instance.root.render(<Dashboard data={data} />);
  return () => {
    if (host.__aiFbiDashboardMount !== instance || generation !== instance.generation) return;
    instance.root.unmount();
    delete host.__aiFbiDashboardMount;
    host.remove();
  };
}
