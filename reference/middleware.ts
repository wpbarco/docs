import { NextRequest, NextResponse } from 'next/server';

/**
 * Middleware to handle case-insensitive redirects for API reference paths.
 * Redirects any path containing uppercase letters to its lowercase equivalent.
 */
export function middleware(request: NextRequest): NextResponse {
  const { pathname, search } = request.nextUrl;

  if (pathname !== pathname.toLowerCase()) {
    const url = request.nextUrl.clone();
    url.pathname = pathname.toLowerCase();

    // Preserve query parameters
    url.search = search;

    // 301 permanent redirect to the lowercase version
    return NextResponse.redirect(url, { status: 301 });
  }

  return NextResponse.next();
}

/**
 * Configure which paths the middleware should run on.
 * This applies to all Python API reference paths.
 */
export const config = {
  matcher: [
    '/python/:path*',
    '/javascript/:path*',
  ],
};