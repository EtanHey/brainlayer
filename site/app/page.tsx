"use client";

import { Nav } from "@/components/nav";
import { Hero } from "@/components/hero";
import { Terminal } from "@/components/terminal";
import { ProblemSolution } from "@/components/problem-solution";
import { SearchExamples } from "@/components/search-examples";
import { Tools } from "@/components/tools";
import { Integrations } from "@/components/integrations";
import { PipelineDemo } from "@/components/pipeline-demo";
import { Cta } from "@/components/cta";
import { Footer } from "@/components/footer";

export default function Home() {
  return (
    <>
      <Nav />
      <Hero />
      <Terminal />
      <Divider />
      <SearchExamples />
      <Divider />
      <div className="section-tinted">
        <ProblemSolution />
      </div>
      <Divider />
      <Tools />
      <Divider />
      <div className="section-tinted">
        <Integrations />
      </div>
      <Divider />
      <PipelineDemo />
      <Cta />
      <Footer />
    </>
  );
}

function Divider() {
  return (
    <div className="mx-auto max-w-[960px] px-6">
      <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent" />
    </div>
  );
}
