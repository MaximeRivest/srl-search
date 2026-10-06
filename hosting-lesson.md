# How to host your own services on one Linux server

*A lesson built on lambda, the server in your house. Written 2026-10-06.*

The goal is to run a public service (like the paper search) on your own machine,
keep it fast when the machine is busy with other work, keep strangers away from
your private files, and make it reachable from the internet without a fixed address.

To do that, you need to understand four layers:

1. **Processes and the kernel:** what actually runs, and who decides what gets the CPU.
2. **systemd:** the program that starts, watches and restarts your services.
3. **cgroups:** how Linux divides CPU, memory and disk between groups of programs.
4. **Networking:** how a request from someone's phone reaches a program in your house.

NixOS sits on top: instead of typing commands to change these settings, you write
them down in a file, and NixOS makes the machine match that file.

Each part ends with **Try it** commands. They only *read* information, unless the
lesson says otherwise, so you can run them safely.

---

## Part 1 — Processes and the kernel

Everything that runs on Linux is a **process**: a program that has been started,
with its own memory, belonging to a **user**. Your terminal is a process. The
embedding job is a process. The web server is a process.

The **kernel** is the core of Linux. Programs can't touch the hardware directly;
they ask the kernel. The kernel decides:

- which process gets a CPU core for the next few milliseconds (the **scheduler**);
- which process gets which part of memory, and what to throw out when memory runs low;
- whose disk reads go first;
- who is allowed to open which file (based on the user).

lambda has 48 CPU threads (a 24-core Threadripper), about 220 GB of RAM and four
SSDs. Dozens of programs share them. Normally the kernel shares fairly: every process
counts the same. **Our goal is to make the sharing unfair on purpose**, in favour of
the public service.

**Try it**

```sh
top                 # live list of processes; press q to quit
ps -u maxime | wc -l   # how many processes run as you
nproc                  # CPU threads the kernel can use
free -g                # memory in GB; "available" is what new work could get
```

### Why "free" memory is usually almost zero

`free -g` on lambda shows ~2 GB "free" but ~183 GB "available". The difference
is the **page cache**: the kernel keeps recently read file contents in spare RAM, so
the next read doesn't touch the disk. It gives that memory back when programs need it.

This matters a lot for search. The search index is a big file on the SSD. Pieces of
it that stay in the page cache are read in nanoseconds; pieces that have been
evicted must come from the SSD, which is much slower. **A fast search server is mostly
about keeping the right parts of the index in the page cache.**

---

## Part 2 — systemd: the manager of services

When lambda boots, the kernel starts one first process: **systemd**. systemd then
starts everything else and keeps watching it. Each thing it manages is a **unit**.
A **service** is a unit for a long-running program.

A service file says things like:

- what command to run;
- what must run first (network, disks);
- what to do if the program crashes (restart after 60 seconds?);
- which user it runs as;
- how much CPU and memory it may use.

Here's a real one from lambda: the embedding job
(`~/Projects/os/machines/hm/openalex-embeddings.nix`), trimmed:

```nix
systemd.user.services.openalex-embeddings = {
  Service = {
    ExecStart = "${root}/.venv/bin/python -u ${root}/supervisor.py";
    Restart = "on-failure";      # crashed? start it again
    RestartSec = 60;             # ...after 60 seconds
    Nice = 10;                   # be polite about CPU (old-style priority)
    MemoryMax = "12G";           # hard ceiling on memory
  };
  Install.WantedBy = [ "default.target" ];   # start it automatically
};
```

You already use the main ideas: auto-start, auto-restart and a memory ceiling.

There are two kinds of systemd:

- **System services** (`systemctl ...`) run for the whole machine. They can run as
  any user, including special locked-down users. Configured in NixOS
  (`machines/hosts/lambda/*.nix`).
- **User services** (`systemctl --user ...`) run *as you*, with access to your files.
  Configured in Home Manager (`machines/hm/*.nix`). The embedding job is one.

**A public service should be a system service**, not a user service. If it were
running as you and someone found a bug in it, they could read everything in your
home folder.

**Try it**

```sh
systemctl --user status openalex-embeddings   # is it running, since when, memory
journalctl --user -u openalex-embeddings -n 20  # its last 20 log lines
systemctl list-units --type=service --state=running | head -30
```

---

## Part 3 — cgroups: dividing the machine

### The idea

A **cgroup** (control group) is a box of processes. The kernel can give each box
rules: "this box gets a large share of CPU", "this box may never use more than 12 GB",
"this box's disk reads are limited to 500 MB/s".

Boxes go inside boxes, like folders. systemd builds this tree for you: every service
gets its own box. Groups of boxes are called **slices**.

lambda's tree looks like this (simplified):

```text
-.slice                    ← the whole machine
├── system.slice           ← system services (vLLM models, Caddy, tailscaled…)
│   ├── caddy.service
│   └── model-vllm-....service
└── user.slice             ← people
    └── user-1000.slice    ← you (maxime)
        └── user@1000.service
            └── app.slice
                └── openalex-embeddings.service
```

**Try it**

```sh
systemd-cgls --no-pager | less     # the real tree with process numbers
systemd-cgtop                      # live: which box uses the most CPU, memory, disk
```

### The most important rule: shares are compared between siblings

CPU shares only compete with **boxes at the same level**. If `public.slice` has a
weight of 1000 and sits next to `system.slice` (100) and `user.slice` (100), then when
all three want the CPU at once, the split is:

```text
public 1000 / (1000+100+100) ≈ 83%   system ≈ 8%   user ≈ 8%
```

Whatever happens *inside* `user.slice` (your desktop, the embedding job, notebooks)
shares that one 8%. Placement in the tree matters as much as the numbers.

**When nobody else wants the CPU, a box may use all of it**, whatever its weight.
Weights only matter when there is competition. That's why they're the best tool: no
CPU sits idle, and the public service still wins whenever it needs to.

### The controls, in plain words

**CPU**

| Setting | Meaning | Kind |
|---|---|---|
| `CPUWeight=1000` | "When there's a queue, I count 10× more than the default (100)." | Priority |
| `CPUWeight=idle` | "Only use CPU nobody else wants." | Priority |
| `CPUQuota=400%` | "Never more than 4 cores' worth, even if the rest is idle." | Wall |
| `AllowedCPUs=0-3` | "Only run on these cores." Combine with keeping others off them to reserve cores. | Wall |

**I tested this on lambda.** Two busy loops forced onto the same CPU thread, one with
weight 1000 and one with weight 100. After 8 seconds the first had used **7.28 s** of
CPU and the second **0.74 s**: almost exactly 10 to 1.

**Memory**

| Setting | Meaning | Kind |
|---|---|---|
| `MemoryMin=8G` | "Never take these 8 GB from me." | Hard protection |
| `MemoryLow=24G` | "Take my first 24 GB only if there's no other choice." | Soft protection |
| `MemoryHigh=100G` | "Above this, slow me down and squeeze my cache." | Soft wall |
| `MemoryMax=12G` | "Never above this; kill me if I can't shrink." | Hard wall |

The key point for search: **the page cache counts as the box's memory.** So
`MemoryLow=24G` on the search service also protects *up to 24 GB of the index that's
cached in RAM*. Without it, a background job reading huge files can push the index
out of RAM, and the next searches suddenly have to wait for the SSD.

A real example from today: the embedding job has hit its 12 GB ceiling **36,286
times**, even though the program itself only uses about 2 GB. It reads large data
files, the kernel caches them, the cache fills the box to 12 GB, and the kernel throws
some cache away to stay under. That's the wall working as intended: the job pays a
small cost, and the rest of the machine is protected.

```sh
cat /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/openalex-embeddings.service/memory.events
#  max 36286   ← times it touched MemoryMax
#  oom_kill 0  ← times the kernel had to kill it (none: good)
```

**Disk**

| Setting | Meaning | Kind |
|---|---|---|
| `IOWeight=1000` | Disk priority, like `CPUWeight`. | Priority |
| `IOReadBandwidthMax=/dev/nvme0n1 500M` | Read at most 500 MB/s from that disk. | Wall |
| `IOReadIOPSMax=/dev/nvme0n1 20000` | At most 20,000 read requests per second. | Wall |

**Catch:** `IOWeight` only works if the disk uses a scheduler that understands weights.
lambda's big NVMe uses `none`:

```sh
cat /sys/block/nvme0n1/queue/scheduler     # [none] mq-deadline kyber
```

`none` is the fastest choice for NVMe, but it ignores weights. So on lambda, **use
bandwidth/IOPS limits on background jobs** if they slow down searches. The other option
is switching to the BFQ scheduler, which costs some speed on fast drives.

**GPU:** cgroups can't share a GPU's computing power. The only reliable wall is
"this service gets GPU 0, background jobs get GPU 1", which is what the embedding job
already does. A CPU-only search service avoids the problem.

### How to know whether the walls are needed: pressure

Linux measures how much time programs spend **waiting** for CPU, memory or disk. This is
called **PSI** (pressure stall information).

```sh
cat /proc/pressure/cpu
cat /proc/pressure/memory
cat /proc/pressure/io
# some avg10=0.01   ← % of the last 10 s where at least one program was waiting
# full avg10=0.00   ← % where every busy program was waiting (worse)
```

Each box has its own: `cat /sys/fs/cgroup/<path>/memory.pressure`.
**If the search service's pressure numbers stay near zero while background jobs run,
your rules are working.** If they rise, tighten the walls. Measure first, then add walls.

### Experiment yourself (this one does run something)

This repeats my test. It uses one CPU thread for about 10 seconds, then cleans up.

```sh
systemd-run --user -q -p CPUWeight=1000 --unit=lesson-fast taskset -c 5 sh -c 'while :; do :; done'
systemd-run --user -q -p CPUWeight=100  --unit=lesson-slow taskset -c 5 sh -c 'while :; do :; done'
sleep 8
systemctl --user show lesson-fast lesson-slow -p CPUUsageNSec   # nanoseconds of CPU each got
systemctl --user stop lesson-fast lesson-slow
```

`systemd-run` creates a temporary box with rules, without editing any file.
`taskset -c 5` pins both loops to CPU thread 5 so they have to compete. Try changing
the weights, or try `-p CPUQuota=20%`.

---

## Part 4 — NixOS: writing the rules down

On most Linux systems you'd edit files in `/etc` by hand, and a year later nobody
remembers why. On NixOS the whole machine is described in `~/Projects/os/machines`.
You change the description, run `~/Projects/os/deploy`, and the machine is rebuilt to
match. If something breaks, you choose the previous version at boot.

Here's what a public search service on lambda could look like. **This is an example
for learning. It isn't installed.**

```nix
# machines/hosts/lambda/public-search.nix
{ ... }:
{
  # A top-level box for things strangers use. Sibling of system.slice and user.slice.
  systemd.slices.public = {
    description = "Services used by the public";
    sliceConfig = {
      CPUWeight = 1000;     # wins CPU competitions ~10 to 1
      MemoryLow = "24G";    # keep up to 24 GB (including cached index) in RAM
    };
  };

  # A box for heavy work that can wait.
  systemd.slices.background = {
    description = "Batch jobs that can wait";
    sliceConfig = {
      CPUWeight = 20;
      MemoryHigh = "120G";  # beyond this: slowed down, cache squeezed
      MemoryMax = "150G";   # never beyond this
      IOReadBandwidthMax = "/dev/nvme0n1 1500M";
    };
  };

  systemd.services.paper-search = {
    description = "Paper search API";
    wantedBy = [ "multi-user.target" ];          # start at boot
    after = [ "network-online.target" ];
    wants = [ "network-online.target" ];
    serviceConfig = {
      Slice = "public.slice";                    # put it in the protected box
      ExecStart = "/path/to/search-server --listen 127.0.0.1:8080";
      Restart = "always";
      RestartSec = 2;

      # --- sandbox: see Part 5 ---
      DynamicUser = true;
      ProtectHome = true;
      ProtectSystem = "strict";
      PrivateTmp = true;
      NoNewPrivileges = true;
      ReadOnlyPaths = [ "/mnt/fast/search-index" ];
      StateDirectory = "paper-search";           # its own writable folder, /var/lib/paper-search
      RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
    };
  };

  # Example: move a heavy system service into the background box.
  # systemd.services.some-batch-job.serviceConfig.Slice = "background.slice";
}
```

After deploying, check that the rules took effect:

```sh
systemctl show paper-search -p Slice -p CPUWeight -p MemoryLow
systemctl show public.slice -p CPUWeight -p MemoryLow
systemd-cgtop
```

**A detail about your own jobs.** Home Manager user services, like the embedding
job, live under `user.slice`, so they can't be moved into `background.slice`. They're
already handled by the sibling rule: all of `user.slice` competes as one box against
`public.slice`'s weight of 1000. To push them further back, give them a low
`CPUWeight` inside their own unit, or move heavy jobs to system services.

**Trade-off of `MemoryLow`:** memory protected for the search service is memory other
jobs can't count on. With 220 GB that's affordable; on a 32 GB server you'd pick the
number carefully.

---

## Part 5 — Walls for safety, not just speed

Resource rules stop the public service from being *slowed down*. Sandbox rules limit
the damage if someone *attacks* it. The second matters more on a family computer.

| Setting | What it does |
|---|---|
| `DynamicUser=true` | Runs as a brand-new, temporary user with no rights to anyone's files. |
| `ProtectHome=true` | `/home` (everyone's personal files) looks empty to the service. |
| `ProtectSystem="strict"` | The whole system is read-only to it. |
| `ReadOnlyPaths=` | Lets it read the index but not change it. |
| `StateDirectory=` | The one folder it may write to. |
| `NoNewPrivileges=true` | It can never gain more rights, even through a bug. |
| `PrivateTmp=true` | Its own temporary folder, invisible to others. |

systemd can grade a service's sandbox:

```sh
systemd-analyze security caddy.service     # a score from 0 (locked) to 10 (exposed)
systemd-analyze security                   # all system services, worst first
```

Another layer: the service listens on `127.0.0.1` only (this machine), never on the
network directly. Only the reverse proxy or tunnel (Part 6) talks to it.

---

## Part 6 — Networking: how a stranger reaches your house

### Addresses

Every device on a network has an **IP address**. There are two kinds:

- **Private** addresses work only inside your home: lambda is `192.168.2.24`.
  Every home in the world reuses these same numbers.
- **Public** addresses are unique on the internet. Your internet provider gives your
  *router* one, and often changes it. That's the "no static IP" problem.

Your router uses **NAT** (network address translation): all devices at home share the
router's single public address. When lambda opens a connection *outward*, the router
remembers it and passes the replies back. But when a stranger tries to connect
*inward*, the router has no idea which device it's for, so it drops the request.

Some providers go further and put many homes behind one shared public address
(**CGNAT**, using addresses `100.64.0.0` to `100.127.255.255`). Then you can't even set
up port forwarding on your router. Fun fact: Tailscale borrows that same address range
for its private network, which is why lambda is also `100.86.49.54`.

**Ports** are like apartment numbers on an address: one machine, many services. Web
pages use port 443 (HTTPS) and 80 (HTTP).

**DNS** turns a name into an address: `search.scholarsreadinglist.com` → some IP.
You control this through your domain registrar (GoDaddy, in your case).

**TLS** (the S in HTTPS) encrypts the connection. It needs a certificate proving you
own the name, usually free from Let's Encrypt. **Whoever holds the certificate can read
the traffic.** Remember this; it decides how sovereign each option below is.

**Try it**

```sh
ip -br addr                   # your addresses: LAN, Tailscale, …
curl -s https://ifconfig.me   # the public address the internet sees
resolvectl query scholarsreadinglist.com   # DNS lookup
```

lambda currently has no public IPv6 address, only local ones. With IPv6 from your
provider, every device could get its own public address and much of the NAT problem
would disappear. Worth asking your provider about.

### The firewall and the reverse proxy

The **firewall** decides which ports accept connections. On NixOS:
`networking.firewall.allowedTCPPorts = [ 80 ];` (see `hosts/lambda/default.nix`).
Everything not listed is blocked.

A **reverse proxy** is a single front door that receives all web requests and sends
each to the right service by name or path. It also handles HTTPS certificates. lambda
already runs one, **Caddy**, for your private portal (`hosts/lambda/portal.nix`). It has
a rule rejecting anyone not on your home network or tailnet:

```text
@outside not remote_ip 127.0.0.0/8 ::1 192.168.2.0/24 100.64.0.0/10 ...
respond @outside "Private portal" 403
```

### Three ways in without a static address

All three work the same basic way: **lambda opens a connection outward**, which NAT
allows, and incoming requests travel back through it. Nothing on your router needs to
be opened.

**A. Cloudflare Tunnel**

```text
user ──HTTPS──▶ Cloudflare ──tunnel──▶ cloudflared on lambda ──▶ 127.0.0.1:8080
```

- Free, simple, uses your own domain, and hides your home address.
- Cloudflare blocks a lot of attack traffic before it reaches you.
- **Not sovereign:** the certificate sits at Cloudflare, so Cloudflare decrypts
  and can see every request. You also depend on their rules and their uptime.
- NixOS has it built in (`services.cloudflared.tunnels`). The tunnel's secret
  credentials file must stay outside git and the Nix store.

**B. Tailscale Funnel**

- You already use Tailscale. Funnel publishes one local service to the internet.
- Encryption ends **on lambda**, so Tailscale's relays only carry encrypted bytes.
- Limits: you get a `*.ts.net` name rather than your own domain, only a few
  ports, and it isn't meant for heavy public traffic.
- Good for a demo or a small number of users.

**C. Your own relay: a small rented server + WireGuard (most sovereign)**

```text
user ──HTTPS──▶ relay (cheap VPS, fixed IP) ══WireGuard tunnel══▶ lambda
                passes encrypted bytes                          Caddy holds the certificate
```

- Rent the smallest server with a fixed public IP (a few euros a month).
- lambda keeps a **WireGuard** tunnel open to it (WireGuard is the encrypted
  tunnel Tailscale is built on).
- The relay just **passes encrypted bytes through**. The certificate and decryption
  stay on lambda, so the relay company can't read the traffic.
- You point your domain's DNS at the relay's fixed IP. Your home address stays hidden.
- Costs: one more machine to keep updated, and **you** handle attack traffic. A
  relay with 1 Gbit/s can still be overwhelmed by a large flood.

| | Own domain | Who can read traffic | Effort | Cost |
|---|---|---|---|---|
| Cloudflare Tunnel | Yes | Cloudflare | Low | Free |
| Tailscale Funnel | No (`ts.net`) | Only you | Lowest | Free |
| Own relay | Yes | Only you | Medium | ~€4–6/month |

**Important:** none of these keeps the service up when lambda or your home internet
is down. They solve *reachability*, not *availability*.

---

## Part 7 — Staying up

| What fails | Protection | NixOS / Linux tool |
|---|---|---|
| Program crashes | Restart automatically | `Restart=always` |
| Machine reboots | Services start by themselves | `wantedBy = [ "multi-user.target" ]` |
| Power cut | Battery (UPS), safe shutdown when low, BIOS "power on after power loss" | `power.ups` (NUT) |
| Disk dies | Two mirrored SSDs + a backup somewhere else | ZFS/btrfs mirror, `services.restic` |
| Internet down | Router with cellular backup | — |
| Nobody notices it's down | Outside check that alerts your phone | Any uptime-check service |
| Everything at home fails | Second copy somewhere else | A rented server |

The habit that matters most: **test the recovery.** Pull the plug on purpose once, and
check that everything comes back by itself.

---

## The order I'd build it in

1. Run the search service as a **system service** in `public.slice`, listening on
   `127.0.0.1` only, with the sandbox settings. Check with `systemd-analyze security`.
2. Run background jobs while sending test searches. Watch `systemd-cgtop` and the
   pressure files. Add walls (memory, disk bandwidth) only where measurements say so.
3. Expose it privately with Tailscale. Then publicly with Funnel for a small test.
4. For real users on your own domain: own relay (sovereign) or Cloudflare Tunnel (easy).
5. Add UPS, monitoring and backups. Test a power cut.

## Words to remember

- **Process:** a running program. **Kernel:** the core of Linux that shares hardware between them.
- **systemd / unit / service:** the manager, and the things it manages.
- **cgroup / slice:** boxes of processes with resource rules. Weights compete between siblings.
- **Page cache:** spare RAM holding recently read files. It counts toward a box's memory.
- **PSI / pressure:** how much time programs spend waiting. The measure of whether walls are needed.
- **NAT / CGNAT:** why strangers can't connect inward to your house.
- **Reverse proxy:** one front door that routes requests and handles HTTPS.
- **Tunnel / relay:** an outward connection that lets requests come back in.
- **TLS certificate:** whoever holds it can read the traffic.
