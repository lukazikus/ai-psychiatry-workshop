// Neural Synapse Background Animation for HTML5 Canvas
(function() {
    const canvas = document.getElementById('synapse-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = canvas.width = canvas.offsetWidth;
    let height = canvas.height = canvas.offsetHeight;

    const nodes = [];
    const maxNodes = 65;
    const connectionDist = 120;
    const mouse = { x: null, y: null, radius: 150 };

    // Track window resize
    window.addEventListener('resize', () => {
        if (!canvas) return;
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
        initNodes();
    });

    // Track mouse movement
    const header = document.querySelector('header');
    if (header) {
        header.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = e.clientX - rect.left;
            mouse.y = e.clientY - rect.top;
        });

        header.addEventListener('mouseleave', () => {
            mouse.x = null;
            mouse.y = null;
        });
    }

    class Node {
        constructor() {
            this.x = Math.random() * width;
            this.y = Math.random() * height;
            this.vx = (Math.random() - 0.5) * 0.6;
            this.vy = (Math.random() - 0.5) * 0.6;
            this.radius = Math.random() * 2 + 1.5;
            this.color = 'rgba(20, 184, 166, 0.8)'; // Electric Teal
        }

        update() {
            // Magnetic pull toward mouse
            if (mouse.x !== null && mouse.y !== null) {
                const dx = mouse.x - this.x;
                const dy = mouse.y - this.y;
                const dist = Math.hypot(dx, dy);
                if (dist < mouse.radius) {
                    const force = (mouse.radius - dist) / mouse.radius;
                    this.vx += (dx / dist) * force * 0.03;
                    this.vy += (dy / dist) * force * 0.03;
                }
            }

            // Apply friction/speed limit
            this.vx = Math.max(Math.min(this.vx, 1.2), -1.2);
            this.vy = Math.max(Math.min(this.vy, 1.2), -1.2);

            this.x += this.vx;
            this.y += this.vy;

            // Bounce off boundaries
            if (this.x < 0 || this.x > width) this.vx *= -1;
            if (this.y < 0 || this.y > height) this.vy *= -1;

            // Clamp coordinates to bounds
            if (this.x < 0) this.x = 0;
            if (this.x > width) this.x = width;
            if (this.y < 0) this.y = 0;
            if (this.y > height) this.y = height;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.shadowBlur = 8;
            ctx.shadowColor = 'rgba(20, 184, 166, 0.5)';
            ctx.fill();
            ctx.shadowBlur = 0; // reset
        }
    }

    function initNodes() {
        nodes.length = 0;
        for (let i = 0; i < maxNodes; i++) {
            nodes.push(new Node());
        }
    }

    function drawLines() {
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.hypot(dx, dy);

                if (dist < connectionDist) {
                    const alpha = (1 - (dist / connectionDist)) * 0.25;
                    ctx.strokeStyle = `rgba(99, 102, 241, ${alpha})`; // Electric Indigo transition lines
                    ctx.lineWidth = 0.8;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.stroke();
                }
            }

            // Draw line to mouse
            if (mouse.x !== null && mouse.y !== null) {
                const dx = nodes[i].x - mouse.x;
                const dy = nodes[i].y - mouse.y;
                const dist = Math.hypot(dx, dy);
                if (dist < mouse.radius) {
                    const alpha = (1 - (dist / mouse.radius)) * 0.35;
                    ctx.strokeStyle = `rgba(20, 184, 166, ${alpha})`; // Electric Teal
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.stroke();
                }
            }
        }
    }

    let isVisible = true;
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            isVisible = entry.isIntersecting;
        });
    });
    observer.observe(canvas);

    function animate() {
        if (isVisible) {
            ctx.clearRect(0, 0, width, height);

            // Subtle base brain mesh network pattern
            drawLines();

            nodes.forEach(node => {
                node.update();
                node.draw();
            });
        }
        requestAnimationFrame(animate);
    }

    initNodes();
    animate();
})();
