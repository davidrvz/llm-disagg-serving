/** @type {import('next').NextConfig} */
const nextConfig = {
  // Router URL injected at build time; override with NEXT_PUBLIC_ROUTER_URL env var
  env: {
    NEXT_PUBLIC_ROUTER_URL: process.env.NEXT_PUBLIC_ROUTER_URL ?? "http://localhost:8000",
  },
};

module.exports = nextConfig;
