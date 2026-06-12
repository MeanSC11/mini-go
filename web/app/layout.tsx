import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "go-bot",
  description: "Play Go against humans or an AlphaZero-style bot",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="topbar">
          <a href="/" className="logo">
            ⚫ go-bot ⚪
          </a>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
