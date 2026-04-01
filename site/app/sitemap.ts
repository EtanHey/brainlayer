import type { MetadataRoute } from "next";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: "https://brainlayer.etanheyman.com",
      changeFrequency: "weekly",
      priority: 1,
    },
    {
      url: "https://brainlayer.etanheyman.com/docs",
      changeFrequency: "weekly",
      priority: 0.8,
    },
  ];
}
