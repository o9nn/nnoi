/**
 * Deep Tree Echo Orchestrator
 *
 * Layered orchestration system for coordinating cognitive services through a unified daemon.
 * Implements Sys6 operadic composition and Dove9 triadic cognitive model.
 *
 * @see https://github.com/o9nn/deltecho
 */

// ============================================================================
// Sys6 Operadic Composition Model
// Sys6 := σ ∘ (φ ∘ μ ∘ (Δ₂ ⊗ Δ₃ ⊗ id_P))
// ============================================================================

class Sys6 {
  constructor() {
    // LCM(2,3,5) = 30-step synchronization clock
    this.SYNC_CYCLE = 30;
    this.STAGES = 5;
    this.STEPS_PER_STAGE = 6;

    // Prime-power delegation lanes
    this.DELTA_2_LANES = 8;  // 2³ = 8-way parallelism
    this.DELTA_3_PHASES = 9; // 3² = 9-phase execution

    // Synchronization state
    this.currentStep = 0;
    this.currentStage = 0;
    this.syncEvents = [];
    this.taskQueues = new Map();

    // Initialize task queues for each lane
    for (let i = 0; i < this.DELTA_2_LANES; i++) {
      this.taskQueues.set(`lane_${i}`, []);
    }
  }

  /**
   * Δ₂ - Prime-power delegation for 8-way parallelism
   * Distributes tasks across 8 parallel execution lanes
   */
  delta2(tasks) {
    const distributed = [];
    for (let i = 0; i < tasks.length; i++) {
      const lane = i % this.DELTA_2_LANES;
      distributed.push({
        ...tasks[i],
        lane,
        parallelGroup: Math.floor(i / this.DELTA_2_LANES)
      });
    }
    return distributed;
  }

  /**
   * Δ₃ - Prime-power delegation for 9-phase execution
   * Assigns phase offsets for coordinated execution
   */
  delta3(tasks) {
    return tasks.map((task, i) => ({
      ...task,
      phase: i % this.DELTA_3_PHASES,
      phaseOffset: (i % this.DELTA_3_PHASES) * (360 / this.DELTA_3_PHASES)
    }));
  }

  /**
   * μ (mu) - LCM synchronization clock
   * Generates sync points for the 30-step cycle
   */
  mu() {
    const syncPoints = [];
    for (let step = 0; step < this.SYNC_CYCLE; step++) {
      // Sync at multiples of 2, 3, and 5
      const syncAt2 = step % 2 === 0;
      const syncAt3 = step % 3 === 0;
      const syncAt5 = step % 5 === 0;

      if (syncAt2 || syncAt3 || syncAt5) {
        syncPoints.push({
          step,
          type: syncAt5 ? 'major' : (syncAt3 ? 'medium' : 'minor'),
          factors: { sync2: syncAt2, sync3: syncAt3, sync5: syncAt5 }
        });
      }
    }
    return syncPoints;
  }

  /**
   * φ (phi) - Compression function (2×3 → 4 reduction)
   * Reduces 6 parallel results into 4 compressed outputs
   */
  phi(results) {
    if (results.length <= 4) return results;

    const compressed = [];
    const groupSize = Math.ceil(results.length / 4);

    for (let i = 0; i < 4; i++) {
      const group = results.slice(i * groupSize, (i + 1) * groupSize);
      compressed.push({
        groupIndex: i,
        merged: group,
        summary: this._mergeResults(group)
      });
    }
    return compressed;
  }

  /**
   * σ (sigma) - Stage scheduler
   * Manages 5 stages × 6 steps = 30-cycle execution
   */
  sigma(task) {
    const stage = this.currentStage;
    const step = this.currentStep;

    const scheduled = {
      ...task,
      stage,
      step,
      cyclePosition: stage * this.STEPS_PER_STAGE + step,
      scheduledAt: Date.now()
    };

    // Advance step counter
    this.currentStep = (this.currentStep + 1) % this.STEPS_PER_STAGE;
    if (this.currentStep === 0) {
      this.currentStage = (this.currentStage + 1) % this.STAGES;
    }

    return scheduled;
  }

  /**
   * Compose the full Sys6 pipeline
   * Sys6 := σ ∘ (φ ∘ μ ∘ (Δ₂ ⊗ Δ₃ ⊗ id_P))
   */
  compose(tasks) {
    // Apply Δ₂ ⊗ Δ₃ (tensor product of prime delegations)
    const parallelized = this.delta2(tasks);
    const phased = this.delta3(parallelized);

    // Apply μ (synchronization)
    const syncPoints = this.mu();
    const synchronized = phased.map(task => ({
      ...task,
      syncPoints: syncPoints.filter(sp =>
        sp.step % (task.lane + 1) === 0
      )
    }));

    // Apply φ (compression)
    const compressed = this.phi(synchronized);

    // Apply σ (scheduling)
    const scheduled = compressed.map(group => this.sigma(group));

    return {
      tasks: scheduled,
      syncEvents: this._generateSyncEvents(scheduled),
      cycleInfo: {
        totalSteps: this.SYNC_CYCLE,
        stages: this.STAGES,
        stepsPerStage: this.STEPS_PER_STAGE,
        syncEventsPerCycle: 42 // 42 synchronization events per 30-step cycle
      }
    };
  }

  _mergeResults(group) {
    return group.reduce((acc, item) => {
      return { ...acc, ...item };
    }, {});
  }

  _generateSyncEvents(scheduled) {
    const events = [];
    scheduled.forEach((task, i) => {
      if (task.syncPoints) {
        task.syncPoints.forEach(sp => {
          events.push({
            taskIndex: i,
            ...sp,
            timestamp: Date.now() + sp.step * 100
          });
        });
      }
    });
    return events;
  }
}

// ============================================================================
// Dove9 Triadic Cognitive Model
// Three concurrent streams at 120° phase offset: SENSE, PROCESS, ACT
// ============================================================================

class Dove9 {
  constructor() {
    // 12-step cycle with 3 streams at 120° offset
    this.CYCLE_LENGTH = 12;
    this.PHASE_OFFSET = 120; // degrees

    // Stream definitions
    this.streams = {
      SENSE: { phase: 0, queue: [], state: 'idle' },
      PROCESS: { phase: 120, queue: [], state: 'idle' },
      ACT: { phase: 240, queue: [], state: 'idle' }
    };

    // Shared salience landscape
    this.salienceMap = new Map();

    // Feedback/feedforward buffers
    this.feedbackBuffer = [];
    this.feedforwardBuffer = [];

    this.currentStep = 0;
  }

  /**
   * SENSE stream - Input gathering and perception
   */
  sense(input) {
    const senseResult = {
      type: 'SENSE',
      timestamp: Date.now(),
      step: this.currentStep,
      phase: this.streams.SENSE.phase,
      data: input,
      salience: this._computeSalience(input)
    };

    this.streams.SENSE.queue.push(senseResult);
    this._updateSalienceMap(senseResult);

    // Feedforward to PROCESS
    this.feedforwardBuffer.push({
      from: 'SENSE',
      to: 'PROCESS',
      data: senseResult
    });

    return senseResult;
  }

  /**
   * PROCESS stream - Cognitive processing and reasoning
   */
  process(input) {
    const processResult = {
      type: 'PROCESS',
      timestamp: Date.now(),
      step: this.currentStep,
      phase: this.streams.PROCESS.phase,
      input: input,
      reasoning: this._applyReasoning(input),
      decisions: this._makeDecisions(input)
    };

    this.streams.PROCESS.queue.push(processResult);

    // Feedback to SENSE for loop refinement
    this.feedbackBuffer.push({
      from: 'PROCESS',
      to: 'SENSE',
      adjustment: processResult.reasoning.salienceAdjustment
    });

    // Feedforward to ACT
    this.feedforwardBuffer.push({
      from: 'PROCESS',
      to: 'ACT',
      data: processResult
    });

    return processResult;
  }

  /**
   * ACT stream - Action execution and output
   */
  act(input) {
    const actResult = {
      type: 'ACT',
      timestamp: Date.now(),
      step: this.currentStep,
      phase: this.streams.ACT.phase,
      input: input,
      actions: this._executeActions(input),
      effects: []
    };

    this.streams.ACT.queue.push(actResult);

    // Feedback to PROCESS for learning
    this.feedbackBuffer.push({
      from: 'ACT',
      to: 'PROCESS',
      outcomes: actResult.actions
    });

    return actResult;
  }

  /**
   * Run one step of the triadic loop
   */
  tick() {
    const stepInCycle = this.currentStep % this.CYCLE_LENGTH;
    const results = {};

    // Determine which streams are active based on phase
    const activePhase = (stepInCycle * 30) % 360; // 30° per step

    // SENSE stream (0° offset)
    if (this._isStreamActive('SENSE', activePhase)) {
      const senseInput = this.feedbackBuffer
        .filter(f => f.to === 'SENSE')
        .pop();
      results.sense = this.sense(senseInput?.adjustment || {});
    }

    // PROCESS stream (120° offset)
    if (this._isStreamActive('PROCESS', activePhase)) {
      const processInput = this.feedforwardBuffer
        .filter(f => f.to === 'PROCESS')
        .pop();
      results.process = this.process(processInput?.data || {});
    }

    // ACT stream (240° offset)
    if (this._isStreamActive('ACT', activePhase)) {
      const actInput = this.feedforwardBuffer
        .filter(f => f.to === 'ACT')
        .pop();
      results.act = this.act(actInput?.data || {});
    }

    this.currentStep++;
    return results;
  }

  /**
   * Run a complete 12-step cycle
   */
  runCycle() {
    const cycleResults = [];
    for (let i = 0; i < this.CYCLE_LENGTH; i++) {
      cycleResults.push(this.tick());
    }
    return {
      cycle: Math.floor(this.currentStep / this.CYCLE_LENGTH),
      steps: cycleResults,
      salienceSnapshot: Object.fromEntries(this.salienceMap)
    };
  }

  _isStreamActive(streamName, currentPhase) {
    const streamPhase = this.streams[streamName].phase;
    const phaseDiff = Math.abs(currentPhase - streamPhase);
    return phaseDiff < 60 || phaseDiff > 300; // Active within ±60° window
  }

  _computeSalience(input) {
    // Compute attention salience based on input properties
    const base = 0.5;
    const novelty = input?.isNew ? 0.3 : 0;
    const urgency = input?.priority === 'high' ? 0.2 : 0;
    return Math.min(1.0, base + novelty + urgency);
  }

  _updateSalienceMap(senseResult) {
    const key = senseResult.data?.id || `sense_${senseResult.timestamp}`;
    this.salienceMap.set(key, {
      salience: senseResult.salience,
      lastUpdate: senseResult.timestamp
    });
  }

  _applyReasoning(input) {
    return {
      analyzed: true,
      patterns: [],
      salienceAdjustment: input?.salience ? input.salience * 0.1 : 0,
      confidence: 0.8
    };
  }

  _makeDecisions(input) {
    return {
      shouldAct: input?.reasoning?.confidence > 0.5,
      priority: input?.salience > 0.7 ? 'high' : 'normal',
      actions: []
    };
  }

  _executeActions(input) {
    const actions = input?.decisions?.actions || [];
    return actions.map(action => ({
      ...action,
      executed: true,
      executedAt: Date.now()
    }));
  }
}

// ============================================================================
// Service Interfaces
// ============================================================================

/**
 * IPC Server - Inter-process communication for local service coordination
 */
class IPCServer {
  constructor() {
    this.handlers = new Map();
    this.connections = new Map();
    this.messageQueue = [];
  }

  register(channel, handler) {
    this.handlers.set(channel, handler);
  }

  send(channel, message) {
    const handler = this.handlers.get(channel);
    if (handler) {
      return handler(message);
    }
    this.messageQueue.push({ channel, message, timestamp: Date.now() });
    return null;
  }

  connect(clientId, options = {}) {
    this.connections.set(clientId, {
      connected: true,
      connectedAt: Date.now(),
      ...options
    });
    return { clientId, status: 'connected' };
  }

  disconnect(clientId) {
    this.connections.delete(clientId);
    return { clientId, status: 'disconnected' };
  }

  broadcast(message) {
    const results = [];
    this.connections.forEach((_, clientId) => {
      results.push({ clientId, delivered: true });
    });
    return results;
  }
}

/**
 * Webhook Server - Event-driven integration with external systems
 */
class WebhookServer {
  constructor() {
    this.endpoints = new Map();
    this.eventLog = [];
    this.retryQueue = [];
    this.maxRetries = 3;
  }

  registerEndpoint(name, config) {
    this.endpoints.set(name, {
      ...config,
      registeredAt: Date.now(),
      callCount: 0
    });
  }

  async emit(eventName, payload) {
    const event = {
      id: `evt_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      name: eventName,
      payload,
      timestamp: Date.now(),
      delivered: []
    };

    const relevantEndpoints = Array.from(this.endpoints.entries())
      .filter(([_, config]) =>
        !config.events || config.events.includes(eventName)
      );

    for (const [name, config] of relevantEndpoints) {
      try {
        await this._deliverWebhook(name, config, event);
        event.delivered.push({ endpoint: name, success: true });
        config.callCount++;
      } catch (error) {
        event.delivered.push({ endpoint: name, success: false, error: error.message });
        this._scheduleRetry(name, config, event);
      }
    }

    this.eventLog.push(event);
    return event;
  }

  async _deliverWebhook(name, config, event) {
    // Simulate webhook delivery
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        if (config.url && config.url.startsWith('http')) {
          resolve({ status: 200, endpoint: name });
        } else {
          reject(new Error('Invalid webhook URL'));
        }
      }, 10);
    });
  }

  _scheduleRetry(name, config, event) {
    const retryCount = this.retryQueue.filter(
      r => r.name === name && r.eventId === event.id
    ).length;

    if (retryCount < this.maxRetries) {
      this.retryQueue.push({
        name,
        config,
        event,
        eventId: event.id,
        scheduledFor: Date.now() + Math.pow(2, retryCount) * 1000,
        attempt: retryCount + 1
      });
    }
  }

  getEventLog(limit = 100) {
    return this.eventLog.slice(-limit);
  }
}

/**
 * MessageBus - DeltaChat-style secure messaging for inter-agent communication
 */
class MessageBus {
  constructor() {
    this.channels = new Map();
    this.subscribers = new Map();
    this.messageHistory = [];
  }

  createChannel(channelId, options = {}) {
    this.channels.set(channelId, {
      id: channelId,
      createdAt: Date.now(),
      encrypted: options.encrypted || false,
      persistent: options.persistent || false,
      messages: []
    });
    return this.channels.get(channelId);
  }

  subscribe(channelId, subscriberId, callback) {
    if (!this.subscribers.has(channelId)) {
      this.subscribers.set(channelId, new Map());
    }
    this.subscribers.get(channelId).set(subscriberId, callback);
    return { channelId, subscriberId, subscribed: true };
  }

  unsubscribe(channelId, subscriberId) {
    if (this.subscribers.has(channelId)) {
      this.subscribers.get(channelId).delete(subscriberId);
    }
    return { channelId, subscriberId, unsubscribed: true };
  }

  publish(channelId, message) {
    const channel = this.channels.get(channelId);
    if (!channel) {
      throw new Error(`Channel ${channelId} does not exist`);
    }

    const envelope = {
      id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      channelId,
      payload: message,
      timestamp: Date.now(),
      delivered: []
    };

    // Store in channel if persistent
    if (channel.persistent) {
      channel.messages.push(envelope);
    }

    // Deliver to subscribers
    const channelSubs = this.subscribers.get(channelId);
    if (channelSubs) {
      channelSubs.forEach((callback, subscriberId) => {
        try {
          callback(envelope);
          envelope.delivered.push({ subscriberId, success: true });
        } catch (error) {
          envelope.delivered.push({ subscriberId, success: false, error: error.message });
        }
      });
    }

    this.messageHistory.push(envelope);
    return envelope;
  }

  getChannelHistory(channelId, limit = 50) {
    const channel = this.channels.get(channelId);
    if (!channel) return [];
    return channel.messages.slice(-limit);
  }
}

// ============================================================================
// Deep Tree Echo Orchestrator - Main Coordinator
// ============================================================================

class DeepTreeEchoOrchestrator {
  constructor(config = {}) {
    this.config = {
      enableSys6: true,
      enableDove9: true,
      enableIPC: true,
      enableWebhooks: true,
      enableMessageBus: true,
      ...config
    };

    // Initialize components
    this.sys6 = new Sys6();
    this.dove9 = new Dove9();
    this.ipc = new IPCServer();
    this.webhooks = new WebhookServer();
    this.messageBus = new MessageBus();

    // Orchestrator state
    this.state = 'initialized';
    this.taskQueue = [];
    this.results = [];
    this.metrics = {
      tasksProcessed: 0,
      cyclesCompleted: 0,
      eventsEmitted: 0,
      messagesRouted: 0
    };

    // Set up internal message channels
    this._setupInternalChannels();
  }

  _setupInternalChannels() {
    // Create internal orchestration channels
    this.messageBus.createChannel('orchestrator.tasks', { persistent: true });
    this.messageBus.createChannel('orchestrator.events', { persistent: false });
    this.messageBus.createChannel('orchestrator.sync', { persistent: false });

    // Register IPC handlers
    this.ipc.register('task.submit', (msg) => this.submitTask(msg));
    this.ipc.register('task.status', (msg) => this.getTaskStatus(msg.taskId));
    this.ipc.register('orchestrator.status', () => this.getStatus());
  }

  /**
   * Start the orchestrator
   */
  start() {
    this.state = 'running';
    this.webhooks.emit('orchestrator.started', { timestamp: Date.now() });
    return { status: 'started', timestamp: Date.now() };
  }

  /**
   * Stop the orchestrator
   */
  stop() {
    this.state = 'stopped';
    this.webhooks.emit('orchestrator.stopped', { timestamp: Date.now() });
    return { status: 'stopped', timestamp: Date.now() };
  }

  /**
   * Submit a task for orchestrated execution
   */
  submitTask(task) {
    const taskEntry = {
      id: `task_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
      ...task,
      status: 'pending',
      submittedAt: Date.now()
    };

    this.taskQueue.push(taskEntry);
    this.messageBus.publish('orchestrator.tasks', taskEntry);

    return taskEntry;
  }

  /**
   * Process pending tasks through the orchestration pipeline
   */
  processTasks() {
    if (this.taskQueue.length === 0) {
      return { processed: 0, results: [] };
    }

    // Apply Sys6 composition to task queue
    const sys6Result = this.sys6.compose(this.taskQueue);

    // Process through Dove9 cognitive loop
    const cycleResult = this.dove9.runCycle();

    // Update task statuses
    const processedTasks = this.taskQueue.map((task, i) => ({
      ...task,
      status: 'completed',
      sys6Schedule: sys6Result.tasks[i] || null,
      cognitiveResult: cycleResult.steps[i % cycleResult.steps.length],
      completedAt: Date.now()
    }));

    // Store results and clear queue
    this.results.push(...processedTasks);
    const processed = this.taskQueue.length;
    this.taskQueue = [];

    // Update metrics
    this.metrics.tasksProcessed += processed;
    this.metrics.cyclesCompleted++;

    // Emit completion event
    this.webhooks.emit('tasks.processed', { count: processed });
    this.metrics.eventsEmitted++;

    return {
      processed,
      results: processedTasks,
      sys6Info: sys6Result.cycleInfo,
      dove9Cycle: cycleResult.cycle
    };
  }

  /**
   * Get task status by ID
   */
  getTaskStatus(taskId) {
    const pending = this.taskQueue.find(t => t.id === taskId);
    if (pending) return { ...pending, status: 'pending' };

    const completed = this.results.find(t => t.id === taskId);
    if (completed) return completed;

    return { taskId, status: 'not_found' };
  }

  /**
   * Get orchestrator status
   */
  getStatus() {
    return {
      state: this.state,
      metrics: this.metrics,
      pendingTasks: this.taskQueue.length,
      completedTasks: this.results.length,
      sys6: {
        currentStage: this.sys6.currentStage,
        currentStep: this.sys6.currentStep
      },
      dove9: {
        currentStep: this.dove9.currentStep,
        cycle: Math.floor(this.dove9.currentStep / this.dove9.CYCLE_LENGTH)
      },
      services: {
        ipc: { connections: this.ipc.connections.size },
        webhooks: { endpoints: this.webhooks.endpoints.size },
        messageBus: { channels: this.messageBus.channels.size }
      }
    };
  }

  /**
   * Route a message through the orchestrator
   */
  routeMessage(channel, message) {
    this.metrics.messagesRouted++;
    return this.messageBus.publish(channel, message);
  }

  /**
   * Register a webhook endpoint
   */
  registerWebhook(name, config) {
    return this.webhooks.registerEndpoint(name, config);
  }

  /**
   * Subscribe to orchestrator events
   */
  subscribe(channel, subscriberId, callback) {
    return this.messageBus.subscribe(channel, subscriberId, callback);
  }
}

// ============================================================================
// Export to window for Noi integration
// ============================================================================

window.Deltecho = {
  // Core components
  Sys6,
  Dove9,

  // Service interfaces
  IPCServer,
  WebhookServer,
  MessageBus,

  // Main orchestrator
  DeepTreeEchoOrchestrator,

  // Factory function for easy instantiation
  createOrchestrator: (config) => new DeepTreeEchoOrchestrator(config),

  // Version info
  version: '1.0.0',
  repository: 'https://github.com/o9nn/deltecho'
};

// Auto-initialize if configured
if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', () => {
    // Create default orchestrator instance
    if (!window.deltechoOrchestrator) {
      window.deltechoOrchestrator = new DeepTreeEchoOrchestrator();
    }
  });
}
