import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "50 Startups — Regression Dashboard",
  description:
    "Interactive Multiple Linear Regression dashboard for the 50 Startups dataset. Predict startup profit and explore forward feature selection using CRISP-DM methodology.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
